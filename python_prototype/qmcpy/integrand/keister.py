""" Definition for class Keister, a concrete implementation of Integrand """

from numpy import cos, pi

from . import Integrand


class Keister(Integrand):
    """Specify and generate values \
    :math:`f(\mathbf{x}) = \pi^{d/2} \cos(\| \mathbf{x} \|)` for \
    :math:`\mathbf{x} \in \mathbb{R}^d`.
    The standard example integrates the Keister integrand with respect to an \
    IID Gaussian distribution with variance 1/2.
    B. D. Keister, Multidimensional Quadrature Algorithms, \
    `Computers in Physics`, *10*, pp.\ 119-122, 1996.
    """

    def __init__(self, nominal_value=None):
        """
        Initialize Keister Integrand

        Args:
            nominal_value (int): :math:`c` such that :math:`(c, \ldots, c) \in \mathcal{X}`
        """
        super().__init__(nominal_value=nominal_value)

    def g(self, x, coord_index):
        """Original integrand to be integrated if the nominal_value = 0, this is efficient

        Args:
            x: nodes, :math:`\mathbf{x}_{\mathfrak{u},i} = i^{\mathtt{th}}` \
                row of an :math:`n \cdot |\mathfrak{u}|` matrix
            coord_index: set of those coordinates in sequence needed, \
                :math:`\mathfrak{u}`

        Returns:
            :math:`n \cdot p` matrix with values \
            :math:`f(\mathbf{x}_{\mathfrak{u},i},\mathbf{c})` where if \
            :math:`\mathbf{x}_i' = (x_{i, \mathfrak{u}},\mathbf{c})_j`, then \
            :math:`x'_{ij} = x_{ij}` for :math:`j \in \mathfrak{u}`, and \
            :math:`x'_{ij} = c` otherwise
        """

        normx2 = (x ** 2).sum(1)  # ||x||^2
        n_coord_index = len(coord_index)
        if n_coord_index != self.dimension and self.nominal_value != 0:
            normx2 = normx2 + self.nominal_value ** 2 * (
                self.dimension - n_coord_index)
        y = pi ** (n_coord_index / 2) * cos(normx2 ** 0.5)
        return y
