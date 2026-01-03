import numpy as np
import scipy.stats as stats

from ..discrete_distribution.abstract_discrete_distribution import AbstractDiscreteDistribution
from ..util import DimensionError, ParameterError
from .scipy_wrapper import SciPyWrapper

# For doctests: import from the same package base as this module
from ..discrete_distribution import DigitalNetB2


class MultivariateNormalJoint(SciPyWrapper):
    """
    Convenience true measure: joint multivariate normal.

    Example:
    >>> tm = MultivariateNormalJoint(
    ...     sampler=DigitalNetB2(2, seed=7),
    ...     mean=[0.0, 0.0],
    ...     cov=[[1.0, 0.8], [0.8, 1.0]],
    ... )
    >>> tm(4).shape
    (4, 2)
    """

    def __init__(self, sampler, mean=None, cov=None):
        if mean is None:
            mean = [0.0] * sampler.d
        if cov is None:
            cov = np.eye(sampler.d)

        mean = np.asarray(mean, dtype=float)
        cov = np.asarray(cov, dtype=float)

        if mean.ndim != 1:
            raise ParameterError("mean must be a 1D vector.")
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise ParameterError("cov must be a square matrix.")
        if cov.shape[0] != mean.size:
            raise DimensionError("mean and cov dimensions do not match.")

        mvn = stats.multivariate_normal(mean=mean, cov=cov)
        super().__init__(sampler=sampler, scipy_distribs=mvn)
