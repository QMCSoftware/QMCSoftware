""" Definition for class Linear, a concrete implementation of Integrand """

from . import Integrand


class Linear(Integrand):
    """ Specify and generate values :math:`f(\mathbf{x}) = \sum_{i=1}^d x_i` \
    for :math:`\mathbf{x} = (x_1,\ldots,x_d) \in \mathbb{R}^d`."""

    def __init__(self, nominal_value=None):
        """
        Initialize Linear Integrand

        Args:
            nominal_value (int): :math:`c` such that\
                :math:`(c, \ldots, c) \in \mathcal{X}`
        """
        super().__init__(nominal_value=nominal_value)

    def g(self, x, coord_index):
        """
        Original integrand to be integrated

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
        y = x.sum(1)  # Linear sum
        return y
