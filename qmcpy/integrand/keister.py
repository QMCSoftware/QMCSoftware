from ._integrand import Integrand
from ..discrete_distribution import Sobol
from ..true_measure import Gaussian, Lebesgue, BrownianMotion
from ..util import ParameterError
from numpy import *


class Keister(Integrand):
    """
    $f(\\boldsymbol{x}) = \\pi^{d/2} \\cos(\\| \\boldsymbol{x} \\|)$.

    The standard example integrates the Keister integrand with respect to an 
    IID Gaussian distribution with variance 1./2.

    >>> d = 2
    >>> k = Keister(Sobol(d,seed=7))
    >>> x = k.discrete_distrib.gen_samples(2**10)
    >>> y = k.f(x)
    >>> y.mean()
    1.8074379398240916
    >>> y2 = k.f_periodized(x,'baker')
    >>> y2.mean()
    1.808937592602977
    >>> k.set_transform(Gaussian(d,mean=0,covariance=1))
    >>> y3 = k.f(x)
    >>> y3.mean()
    1.808115525723259

    References:

        [1] B. D. Keister, Multidimensional Quadrature Algorithms, 
        `Computers in Physics`, *10*, pp. 119-122, 1996.
    """

    def __init__(self, discrete_distrib):
        """
        Args:
            discrete_distrib (DiscreteDistribution): a discrete distribution instance.
        """
        self.discrete_distrib = discrete_distrib
        self.true_measure = Lebesgue(Gaussian(self.discrete_distrib.d, mean=0, covariance=1/2))
        super(Keister,self).__init__()
    
    def g(self, x):
        norm = sqrt((x**2).sum(1))
        return cos(norm) * exp(-(norm**2))
