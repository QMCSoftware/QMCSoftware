from ._integrand import Integrand
from ..discrete_distribution import Sobol
from ..true_measure import Gaussian, Lebesgue
from ..util import ParameterError
from numpy import *


class Keister(Integrand):
    """
    $f(\\boldsymbol{x}) = \\pi^{d/2} \\cos(\\| \\boldsymbol{x} \\|)$.

    The standard example integrates the Keister integrand with respect to an 
    IID Gaussian distribution with variance 1./2.

    >>> d = 2
    >>> s = Sobol(d,seed=7)
    >>> g = Gaussian(d,covariance=1./2)
    >>> k = Keister(s,g)
    >>> x = s.gen_samples(2**10)
    >>> y = k.f(x)
    >>> y.mean()
    1.80...
    
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
        return cos(normx) * exp(-(normx**2))
