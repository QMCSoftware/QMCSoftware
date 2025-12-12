from .abstract_true_measure import AbstractTrueMeasure
from .uniform import Uniform
from .gaussian import Gaussian
from ..discrete_distribution import DigitalNetB2
from ..util import ParameterError
from scipy.stats import norm
import numpy as np
from typing import Union


class Lebesgue(AbstractTrueMeasure):
    r"""
    Lebesgue measure as described in [https://en.wikipedia.org/wiki/Lebesgue_measure](https://en.wikipedia.org/wiki/Lebesgue_measure).

    Examples:
        >>> Lebesgue(Gaussian(DigitalNetB2(2,seed=7)))
        Lebesgue (AbstractTrueMeasure)
            transform       Gaussian (AbstractTrueMeasure)
                                mean            0
                                covariance      1
                                decomp_type     PCA
        >>> Lebesgue(Uniform(DigitalNetB2(2,seed=7)))
        Lebesgue (AbstractTrueMeasure)
            transform       Uniform (AbstractTrueMeasure)
                                lower_bound     0
                                upper_bound     1
    """
    
    def __init__(self, sampler):
        r"""
        Args:
            sampler (AbstractTrueMeasure): A true measure by which to compose a transform.
        """
        self.parameters = []
        if not isinstance(sampler,AbstractTrueMeasure):
            raise ParameterError("Lebesgue sampler must be an AbstractTrueMeasure by which to transform samples.")
        self.domain = sampler.range # hack to make sure Lebesgue is compatible with any transform
        self.range = sampler.range
        self._parse_sampler(sampler)
        super(Lebesgue,self).__init__()

    def _weight(self, x):
        return np.ones(x.shape[:-1],dtype=float)

    def _spawn(self, sampler, dimension):
        return Lebesgue(sampler)
