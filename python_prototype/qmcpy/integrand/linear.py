""" Definition for class Linear, a concrete implementation of Integrand """

from ._integrand import Integrand


class Linear(Integrand):
    """
    Specify and generate values $f(\\boldsymbol{x}) = \\sum_{i=1}^d x_i$ \
    for $\\boldsymbol{x} = (x_1,\\ldots,x_d) \\in \\mathbb{R}^d$
    """

    def __init__(self, measure):
        """
        Args:
            measure (TrueMeasure): a TrueMeasure instance
        """
        self.measure = measure
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
        y = x.sum(1)  # Linear sum
        return y
