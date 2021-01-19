from ._true_measure import TrueMeasure
from .uniform import Uniform
from .gaussian import Gaussian
from ..discrete_distribution import Sobol
from ..util import TransformError, ParameterError
from scipy.stats import norm
from numpy import *


class Lebesgue(TrueMeasure):
    """
    >>> Lebesgue(Gaussian(Sobol(2,seed=7)))
    Lebesgue (TrueMeasure Object)
        transform       Gaussian (TrueMeasure Object)
                           mean            0
                           covariance      1
                           decomp_type     pca
    >>> Lebesgue(Uniform(Sobol(2,seed=7)))
    Lebesgue (TrueMeasure Object)
        transform       Uniform (TrueMeasure Object)
                           lower_bound     0
                           upper_bound     1
    """
    
    def __init__(self, sampler):
        """
        Args:
            sampler (TrueMeasure): A  true measure by which to compose a transform.
        """
        self.parameters = []
        if not isinstance(sampler,TrueMeasure):
            raise ParameterError("Lebesgue sampler must be a true measure by which to tranform samples.")
        self.domain = sampler.range # hack to make sure Lebesuge is compatible with any tranform
        self.range = None
        self._parse_sampler(sampler)
        super(Lebesgue,self).__init__()

    def _weight(self, x):
        return ones(x.shape[0],dtype=float)

    def _set_dimension(self, dimension):
        self.d = dimension
