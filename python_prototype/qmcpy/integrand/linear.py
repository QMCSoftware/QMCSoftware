""" Definition for class Linear, a concrete implementation of Integrand """

from ._integrand import Integrand


class Linear(Integrand):
    """
    Specify and generate values :math:`f(\\boldsymbol{x}) = \\sum_{i=1}^d x_i` \
    for :math:`\\boldsymbol{x} = (x_1,\\ldots,x_d) \\in \\mathbb{R}^d`
    """

    def __init__(self, measure):
        """
        Args:
            measure (Measure): a Measure instance
        """
        self.measure = measure
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
            :math:`\\boldsymbol{x}_i' = (x_{i, \\mathfrak{u}},\\mathbf{c})_j`, \
            then :math:`x'_{ij} = x_{ij}` for :math:`j \\in \\mathfrak{u}`, \
            and :math:`x'_{ij} = c` otherwise
        """
        y = x.sum(1)  # Linear sum
        return y
