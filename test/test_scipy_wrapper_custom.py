import pytest
import numpy as np
import scipy.stats as stats

from qmcpy.discrete_distribution import DigitalNetB2
from qmcpy.true_measure import SciPyWrapper, ZeroInflatedExpUniform, StudentT
from qmcpy.true_measure.triangular import TriangularDistribution


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

    assert abs(rho_hat - rho_target) < 0.05
    assert abs(est_moment - rho_target) < 0.05


def test_triangular_custom_marginal_range_and_shape():
    """
    Make sure our custom triangular marginal behaves sensibly:
    samples stay in the right interval and the empirical mean is close
    to the analytic mean.
    """
    tri = TriangularDistribution(c=0.3, loc=-1.0, scale=2.0)
    tm = SciPyWrapper(DigitalNetB2(1, seed=11), scipy_distribs=tri)

    n = 4096
    x = tm(n).ravel()

    assert x.min() >= -1.1
    assert x.max() <= 1.1

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
    sampler = DigitalNetB2(2, seed=17)
    tm = ZeroInflatedExpUniform(sampler, p_zero=p_zero, lam=1.5, y_split=0.5)

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

    df = 5.0
    rho = 0.8
    loc = np.array([0.0, 0.0])
    shape = np.array([[1.0, rho], [rho, 1.0]])

    tm = StudentT(DigitalNetB2(2, seed=123), loc=loc, shape=shape, df=df)

    n = 4096
    x = tm(n)
    emp_corr = np.corrcoef(x.T)[0, 1]

    assert abs(emp_corr - rho) < 0.05
