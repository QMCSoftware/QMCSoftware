from ._integrand import Integrand
from numpy import cos, linalg as LA, pi


class Keister(Integrand):
    """
    $f(\\boldsymbol{x}) = \\pi^{d/2} \\cos(\\| \\boldsymbol{x} \\|)$.

    The standard example integrates the Keister integrand with respect to an 
    IID Gaussian distribution with variance 1/2.

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
