""" Definition for class Keister, a concrete implementation of Integrand """

from ._integrand import Integrand

from numpy import cos, linalg as LA, pi


class Keister(Integrand):
    """
    Specify and generate values \
    :math:`f(\\boldsymbol{x}) = \\pi^{d/2} \\cos(\\| \\boldsymbol{x} \\|)` for \
    :math:`\\boldsymbol{x} \\in \\mathbb{R}^d`.

    The standard example integrates the Keister integrand with respect to an \
    IID Gaussian distribution with variance 1/2.

    Reference:
            B. D. Keister, Multidimensional Quadrature Algorithms, \
            `Computers in Physics`, *10*, pp. 119-122, 1996.
    """
    
    def __init__(self, dimension):
        """
        Args:
            dimension (ndarray): dimension(s) of the integrand(s)
        """
        super().__init__(dimension)

    def g(self, x):
        """
        Original integrand to be integrated

        Args:
            x: nodes, :math:`\\boldsymbol{x}_{\\mathfrak{u},i} = i^{\\mathtt{th}}` \
                row of an :math:`n \\cdot |\\mathfrak{u}|` matrix

        Returns:
            :math:`n \\cdot p` matrix with values \
            :math:`f(\\boldsymbol{x}_{\\mathfrak{u},i},\\mathbf{c})` where if \
            :math:`\\boldsymbol{x}_i' = (x_{i, \\mathfrak{u}},\\mathbf{c})_j`, \
            then :math:`x'_{ij} = x_{ij}` for :math:`j \\in \\mathfrak{u}`, \
            and :math:`x'_{ij} = c` otherwise
        """
        dimension = x.shape[1]  # infer domain dimension
        normx = LA.norm(x, 2, axis=1)  # ||x||_2
        y = pi ** (dimension / 2.0) * cos(normx)
        return y
