from ._integrand import Integrand
from ..discrete_distribution import Sobol
from ..true_measure import Gaussian, Lebesgue
from ..discrete_distribution._discrete_distribution import DiscreteDistribution
from ..true_measure._true_measure import TrueMeasure
from ..util import ParameterError
from numpy import *


class Keister(Integrand):
    """
    $f(\\boldsymbol{x}) = \\pi^{d/2} \\cos(\\| \\boldsymbol{x} \\|)$.

    The standard example integrates the Keister integrand with respect to an 
    IID Gaussian distribution with variance 1./2.

    >>> k = Keister(Sobol(2,seed=7))
    >>> x = k.discrete_distrib.gen_samples(2**10)
    >>> y = k.f(x)
    >>> y.mean()
    1.807...
    >>> k.true_measure
    Lebesgue (TrueMeasure Object)
        transform       Gaussian (TrueMeasure Object)
                           mean            0
                           covariance      2^(-1)
                           decomp_type     pca
    >>> k = Keister(Gaussian(Sobol(2,seed=7),mean=0,covariance=2))
    >>> x = k.discrete_distrib.gen_samples(2**10)
    >>> y = k.f(x)
    >>> y.mean()
    1.808...
    >>> yp = k.f_periodized(x,'c2sin')
    >>> yp.mean()
    1.805...

    References:

        [1] B. D. Keister, Multidimensional Quadrature Algorithms, 
        `Computers in Physics`, *10*, pp. 119-122, 1996.
    """

    def __init__(self, sampler):
        """
        Args:
            sampler (DiscreteDistribution/TrueMeasure): A 
                discrete distribution from which to transform samples or a
                true measure by which to compose a transform
        """
        if isinstance(sampler,DiscreteDistribution): # use the default transform
            self.true_measure = Lebesgue(Gaussian(sampler, mean=0, covariance=1/2))
        elif isinstance(sampler,TrueMeasure): # importance sampling
            if (sampler.range!=array([-inf,inf])).any(): 
                raise ParameterError("Keister requires sampler whose transformation range is all reals.")
            self.true_measure = Lebesgue(sampler)
        else:
            raise ParameterError("Keister requires sampler to be a discrete distribution or true measure.")
        super(Keister,self).__init__()
    
    def g(self, x):
        norm = sqrt((x**2).sum(1))
        return cos(norm) * exp(-(norm**2))
