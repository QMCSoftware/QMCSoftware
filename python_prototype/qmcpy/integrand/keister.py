""" Definition for class Keister, a concrete implementation of Integrand """

from numpy import cos, pi

from . import Integrand


class Keister(Integrand):
    """
    Specify and generate values \
    :math:`f(\\boldsymbol{x}) = \\pi^{d/2} \\cos(\| \\boldsymbol{x} \|)` for \
    :math:`\\boldsymbol{x} \\in \\mathbb{R}^d`.

    The standard example integrates the Keister integrand with respect to an \
    IID Gaussian distribution with variance 1/2.

    Reference:
            B. D. Keister, Multidimensional Quadrature Algorithms, \
            `Computers in Physics`, *10*, pp.\ 119-122, 1996.
    """

    def __init__(self):
        """ Initialize Keister Integrand """
        super().__init__()

    def g(self, x):
        """
        Original integrand to be integrated

        Args:
            x: nodes, :math:`\\boldsymbol{x}_{\\mathfrak{u},i} = i^{\\mathtt{th}}` \
                row of an :math:`n \\cdot |\\mathfrak{u}|` matrix

        Returns:
            :math:`n \\cdot p` matrix with values \
            :math:`f(\\boldsymbol{x}_{\\mathfrak{u},i},\\mathbf{c})` where if \
            :math:`\\boldsymbol{x}_i' = (x_{i, \\mathfrak{u}},\\mathbf{c})_j`, then \
            :math:`x'_{ij} = x_{ij}` for :math:`j \\in \\mathfrak{u}`, and \
            :math:`x'_{ij} = c` otherwise
        """
        normx2 = (x ** 2).sum(1)  # ||x||^2
        n_coord_index = x.shape[-1]
        y = pi ** (n_coord_index / 2) * cos(normx2 ** 0.5)
        return y
