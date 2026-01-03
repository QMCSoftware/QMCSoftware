import numpy as np
import scipy.stats as stats
from ..discrete_distribution import DigitalNetB2

from ..util import DimensionError, ParameterError
from .scipy_wrapper import SciPyWrapper


def _split_u01(u, bits=26):
    """
    Split one U(0,1) into two U(0,1) deterministically by bit splitting.
    Lets us generate the extra chi-square variable without increasing dimension.
    """
    u = np.asarray(u, dtype=float)
    one_minus = np.nextafter(1.0, 0.0)
    u = np.clip(u, 0.0, one_minus)

    scale = np.uint64(1 << (2 * bits))
    m = (u * scale).astype(np.uint64)

    a = np.zeros_like(m, dtype=np.uint64)
    b = np.zeros_like(m, dtype=np.uint64)
    for i in range(bits):
        a |= ((m >> (2 * i)) & 1) << i
        b |= ((m >> (2 * i + 1)) & 1) << i

    denom = float(1 << bits)
    return a / denom, b / denom


class _MVTAdapter:
    """
    Joint multivariate Student t adapter with:
      - transform(u): map u in (0,1)^d to R^d
      - logpdf(x): density via scipy.stats.multivariate_t
    """

    def __init__(self, df, mean, shape):
        self.df = float(df)
        if self.df <= 0:
            raise ParameterError("df must be > 0.")

        mean = np.asarray(mean, dtype=float)
        shape = np.asarray(shape, dtype=float)

        if mean.ndim != 1:
            raise ParameterError("mean must be a 1D vector.")
        if shape.ndim != 2 or shape.shape[0] != shape.shape[1]:
            raise ParameterError("shape must be a square matrix.")
        if shape.shape[0] != mean.size:
            raise DimensionError("mean and shape dimensions do not match.")

        self.mean = mean
        self.dim = mean.size
        self._chol = np.linalg.cholesky(shape)

        if not hasattr(stats, "multivariate_t"):
            raise ParameterError("scipy.stats.multivariate_t is required for logpdf.")

        self._dist = stats.multivariate_t(loc=self.mean, shape=shape, df=self.df)

    def transform(self, u):
        u = np.asarray(u, dtype=float)
        if u.shape[-1] != self.dim:
            raise DimensionError(f"Expected last axis {self.dim}, got {u.shape[-1]}")

        eps = np.finfo(float).eps

        u0 = u[..., 0]
        u_chi2, u_norm0 = _split_u01(u0)

        u_norm = np.concatenate([u_norm0[..., None], u[..., 1:]], axis=-1)
        u_norm = np.clip(u_norm, eps, 1.0 - eps)
        u_chi2 = np.clip(u_chi2, eps, 1.0 - eps)

        z = stats.norm.ppf(u_norm)

        z_flat = z.reshape(-1, self.dim)
        z_corr = (z_flat @ self._chol.T).reshape(z.shape)

        v = stats.chi2.ppf(u_chi2, df=self.df)
        scale = np.sqrt(self.df / v)[..., None]

        return self.mean + z_corr * scale

    def logpdf(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape[-1] != self.dim:
            raise DimensionError(f"Expected last axis {self.dim}, got {x.shape[-1]}")

        x_flat = x.reshape(-1, self.dim)
        lp = self._dist.logpdf(x_flat)
        return np.asarray(lp).reshape(x.shape[:-1])


class MultivariateStudentTJoint(SciPyWrapper):
    """
    Convenience true measure: joint multivariate Student t.

    Example:
    >>> tm = MultivariateStudentTJoint(
    ...     sampler=DigitalNetB2(2, seed=7),
    ...     df=5,
    ...     mean=[0.0, 0.0],
    ...     shape=[[1.0, 0.7], [0.7, 1.0]],
    ... )
    >>> tm(4).shape
    (4, 2)
    """

    def __init__(self, sampler, df, mean=None, shape=None):
        d = sampler.d
        if mean is None:
            mean = np.zeros(d)
        if shape is None:
            shape = np.eye(d)

        adapter = _MVTAdapter(df=df, mean=mean, shape=shape)
        super().__init__(sampler=sampler, scipy_distribs=adapter)
