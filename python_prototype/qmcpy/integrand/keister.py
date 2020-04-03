""" Definition for class Keister, a concrete implementation of Integrand """

from ._integrand import Integrand
from numpy import cos, linalg as LA, pi


class Keister(Integrand):
    """
    Specify and generate values \
    $f(\\boldsymbol{x}) = \\pi^{d/2} \\cos(\\| \\boldsymbol{x} \\|)$ for \
    $\\boldsymbol{x} \\in \\mathbb{R}^d$.

    The standard example integrates the Keister integrand with respect to an \
    IID Gaussian distribution with variance 1/2.

    Reference:
            B. D. Keister, Multidimensional Quadrature Algorithms, \
            `Computers in Physics`, *10*, pp. 119-122, 1996.
    """

    def __init__(self, measure):
        """
        Initialize Keister integrand

        Args:
            measure (TrueMeasure): a TrueMeasure instance
        """
        self.measure = measure
        self.dimension = self.measure.dimension
        super().__init__()

    def g(self, x):
        """
        Original integrand to be integrated

        Args:
            x: nodes, $\\boldsymbol{x}_{\\mathfrak{u},i} = i^{\\mathtt{th}}$ \
                row of an $n \\cdot |\\mathfrak{u}|$ matrix

        Returns:
            $n \\cdot p$ matrix with values \
            $f(\\boldsymbol{x}_{\\mathfrak{u},i},\\mathbf{c})$ where if \
            $\\boldsymbol{x}_i' = (x_{i, \\mathfrak{u}},\\mathbf{c})_j$, \
            then $x'_{ij} = x_{ij}$ for $j \\in \\mathfrak{u}$, \
            and $x'_{ij} = c$ otherwise

        """
        normx = LA.norm(x, 2, axis=1)  # ||x||_2
        y = pi ** (self.dimension / 2.0) * cos(normx)
        return y
