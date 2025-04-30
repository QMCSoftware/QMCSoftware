from ._true_measure import TrueMeasure
from .uniform import Uniform
from .gaussian import Gaussian
from ..discrete_distribution import DigitalNetB2
from ..util import TransformError, ParameterError
from scipy.stats import norm
import numpy as np


class Lebesgue(TrueMeasure):
    """
    >>> Lebesgue(Gaussian(DigitalNetB2(2,seed=7)))
    Lebesgue (TrueMeasure Object)
        transform       Gaussian (TrueMeasure Object)
                           mean            0
                           covariance      1
                           decomp_type     PCA
    >>> Lebesgue(Uniform(DigitalNetB2(2,seed=7)))
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
            raise ParameterError("Lebesgue sampler must be a true measure by which to transform samples.")
        self.domain = sampler.range # hack to make sure Lebesgue is compatible with any transform
        self.range = sampler.range
        self._parse_sampler(sampler)
        super(Lebesgue,self).__init__()

    def _weight(self, x):
        return np.ones(x.shape[0],dtype=float)

    def _spawn(self, sampler, dimension):
        return Lebesgue(sampler)
