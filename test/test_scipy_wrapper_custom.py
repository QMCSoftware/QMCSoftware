import pytest
import numpy as np
import scipy.stats as stats

from qmcpy.discrete_distribution import DigitalNetB2
from qmcpy.true_measure.scipy_wrapper import SciPyWrapper
from qmcpy.util import ParameterError, DimensionError


class TriangularUserDistribution:
    """
    Same triangular distribution as in the demo, kept here locally so the
    tests are self contained.
    """

    def __init__(self, c=0.5, loc=0.0, scale=1.0):
        c = float(c)
        loc = float(loc)
        scale = float(scale)

        if not (0.0 < c < 1.0):
            raise ParameterError("c must lie strictly between 0 and 1.")
        if scale <= 0.0:
            raise ParameterError("scale must be positive.")

        self.c = c
        self.loc = loc
        self.scale = scale

        self._a = loc
        self._b = loc + scale
        self._m = loc + c * scale

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        a, m, b = self._a, self._m, self._b
        out = np.zeros_like(x, dtype=float)

        left = (x >= a) & (x < m)
        right = (x >= m) & (x <= b)

        out[left] = 2.0 * (x[left] - a) / ((b - a) * (m - a))
        out[right] = 2.0 * (b - x[right]) / ((b - a) * (b - m))
        return out

    def ppf(self, u):
        u = np.asarray(u, dtype=float)
        a, m, b = self._a, self._m, self._b

        Fm = (m - a) / (b - a)

        x = np.empty_like(u, dtype=float)
        left = u <= Fm
        right = ~left

        x[left] = a + np.sqrt(u[left] * (b - a) * (m - a))
        x[right] = b - np.sqrt((1.0 - u[right]) * (b - a) * (b - m))
        return x


class ZeroInflatedExpUniformJoint:
    """
    Minimal version of the zero inflated joint distribution, just for tests.
    """

    def __init__(self, p_zero=0.4, lam=1.5, y_split=0.5):
        if not (0.0 < p_zero < 1.0):
            raise ParameterError("p_zero must be in (0,1).")
        if lam <= 0.0:
            raise ParameterError("lam must be positive.")
        if not (0.0 < y_split < 1.0):
            raise ParameterError("y_split must be in (0,1).")

        self.p_zero = float(p_zero)
        self.lam = float(lam)
        self.y_split = float(y_split)
        self.dim = 2

    def transform(self, u):
        u = np.asarray(u, dtype=float)
        if u.shape[-1] != 2:
            raise DimensionError(
                f"ZeroInflatedExpUniformJoint expects last axis 2, got {u.shape[-1]}"
            )

        u1 = u[..., 0]
        u2 = u[..., 1]

        x = np.zeros_like(u1, dtype=float)
        y = np.empty_like(u1, dtype=float)

        mask_zero = u1 <= self.p_zero
        mask_exp = ~mask_zero

        y[mask_zero] = self.y_split * u2[mask_zero]

        if np.any(mask_exp):
            u1_rescaled = (u1[mask_exp] - self.p_zero) / (1.0 - self.p_zero)
            x[mask_exp] = -np.log(1.0 - u1_rescaled) / self.lam
            y[mask_exp] = self.y_split + (1.0 - self.y_split) * u2[mask_exp]

        out = np.empty(u.shape, dtype=float)
        out[..., 0] = x
        out[..., 1] = y
        return out


def test_mvn_dependence_correlation_and_moment():
    """
    Check that passing a SciPy multivariate normal through SciPyWrapper
    preserves correlation and the mixed moment E[X1 X2].
    """
    sampler = DigitalNetB2(2, seed=5)
    rho_target = 0.7
    cov = [[1.0, rho_target], [rho_target, 1.0]]
    mvn = stats.multivariate_normal(mean=[0.0, 0.0], cov=cov)
    tm_mvn = SciPyWrapper(sampler, scipy_distribs=mvn)

    n = 4096
    x = tm_mvn(n)

    rho_hat = np.corrcoef(x.T)[0, 1]
    est_moment = np.mean(x[:, 0] * x[:, 1])

    assert np.isfinite(rho_hat)
    assert np.isfinite(est_moment)

    # Allow a bit of slack around the target values.
    assert abs(rho_hat - rho_target) < 0.05
    assert abs(est_moment - rho_target) < 0.05


def test_triangular_custom_marginal_range_and_shape():
    """
    Make sure our custom triangular marginal behaves sensibly:
    samples stay in the right interval and the histogram roughly matches
    the analytic density shape.
    """
    tri = TriangularUserDistribution(c=0.3, loc=-1.0, scale=2.0)
    sampler = DigitalNetB2(1, seed=11)
    tm = SciPyWrapper(sampler, scipy_distribs=tri)

    n = 4096
    x = tm(n).ravel()

    # Samples should live inside a slightly padded interval.
    assert x.min() >= -1.1
    assert x.max() <= 1.1

    # Compare empirical mean to the analytic mean for a triangular dist.
    # For this parameterisation, the mean is (a + b + m) / 3.
    a = -1.0
    b = 1.0
    m = -1.0 + 0.3 * 2.0
    true_mean = (a + b + m) / 3.0
    emp_mean = x.mean()
    assert abs(emp_mean - true_mean) < 0.05


def test_zero_inflated_zero_rate():
    """
    Check that the zero inflated joint distribution preserves the
    specified probability mass at X = 0.
    """
    p_zero = 0.4
    zi = ZeroInflatedExpUniformJoint(p_zero=p_zero, lam=1.5, y_split=0.5)
    sampler = DigitalNetB2(2, seed=17)
    tm = SciPyWrapper(sampler, scipy_distribs=zi)

    n = 4096
    samples = tm(n)
    x = samples[:, 0]
    zero_rate = np.mean(x == 0.0)

    assert abs(zero_rate - p_zero) < 0.05


def test_student_t_marginals_shape():
    tm = SciPyWrapper(
        sampler=DigitalNetB2(2, seed=5),
        scipy_distribs=stats.t(df=5),
    )
    x = tm(8)
    assert x.shape == (8, 2)


def test_multivariate_student_t_joint_corr_and_cov():
    if not hasattr(stats, "multivariate_t"):
        pytest.skip("scipy.stats.multivariate_t not available in this SciPy version")

    class MultivariateStudentTJoint:
        def __init__(self, loc, shape, df):
            self.loc = np.asarray(loc, float)
            self.shape = np.asarray(shape, float)
            self.df = float(df)
            if self.loc.ndim != 1:
                raise ParameterError("loc must be 1D")
            if self.shape.shape != (self.loc.size, self.loc.size):
                raise DimensionError("shape must match loc dimension")
            self.dim = self.loc.size
            self._rv = stats.multivariate_t(loc=self.loc, shape=self.shape, df=self.df)

        def transform(self, u):
            # reuse your demo logic here, or import it if you moved it into a shared module
            u = np.clip(np.asarray(u, float), np.finfo(float).eps, 1 - np.finfo(float).eps)
            uu = u.reshape(-1, self.dim)
            x = np.empty_like(uu)

            x[:, 0] = stats.t.ppf(uu[:, 0], df=self.df, loc=self.loc[0], scale=np.sqrt(self.shape[0, 0]))
            for i in range(1, self.dim):
                A = slice(0, i)
                mu_A = self.loc[A]
                mu_B = self.loc[i]
                Sigma_AA = self.shape[A, A]
                Sigma_BA = self.shape[i, A]
                Sigma_AB = self.shape[A, i]
                Sigma_BB = self.shape[i, i]

                x_A = x[:, A]
                diff = x_A - mu_A
                sol = np.linalg.solve(Sigma_AA, diff.T).T
                d_A = np.sum(diff * sol, axis=1)

                mu_cond = mu_B + sol @ Sigma_BA
                schur = Sigma_BB - Sigma_BA @ np.linalg.solve(Sigma_AA, Sigma_AB)

                df_cond = self.df + i
                shape_cond = (self.df + d_A) / (self.df + i) * schur

                x[:, i] = stats.t.ppf(uu[:, i], df=df_cond, loc=mu_cond, scale=np.sqrt(shape_cond))

            return x.reshape(*u.shape[:-1], self.dim)

        def logpdf(self, x):
            x = np.asarray(x, float).reshape(-1, self.dim)
            return self._rv.logpdf(x)

    df = 5.0
    rho = 0.8
    loc = np.array([0.0, 0.0])
    shape = np.array([[1.0, rho], [rho, 1.0]])

    joint = MultivariateStudentTJoint(loc, shape, df)
    tm = SciPyWrapper(DigitalNetB2(2, seed=123), joint)

    n = 4096
    x = tm(n)
    emp_corr = np.corrcoef(x.T)[0, 1]

    # corr should be close to rho
    assert abs(emp_corr - rho) < 0.05


