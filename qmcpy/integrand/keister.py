from ._integrand import Integrand
from ..discrete_distribution import Sobol
from ..true_measure import Gaussian
from numpy import cos, linalg as LA, pi


class Keister(Integrand):
    """
    $f(\\boldsymbol{x}) = \\pi^{d/2} \\cos(\\| \\boldsymbol{x} \\|)$.

    The standard example integrates the Keister integrand with respect to an 
    IID Gaussian distribution with variance 1/2.

    >>> dd = Sobol(2,seed=7)
    >>> m = Gaussian(dd,covariance=1./2)
    >>> k = Keister(m)
    >>> x = dd.gen_samples(2**10)
    >>> y = k.f(x)
    >>> y.mean()
    1.8082479629092816
    
    Reference
        B. D. Keister, Multidimensional Quadrature Algorithms, 
        `Computers in Physics`, *10*, pp. 119-122, 1996.
    """

    def __init__(self, measure):
        """
        Args:
            measure (TrueMeasure): a TrueMeasure instance
        """
        self.measure = measure
        self.dimension = self.measure.dimension
        super().__init__()

    def g(self, x):
        """ See abstract method. """
        normx = LA.norm(x, 2, axis=1)  # ||x||_2
        y = pi ** (self.dimension / 2.0) * cos(normx)
        return y
