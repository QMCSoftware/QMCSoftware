""" Definition for class QuickConstruct, a concrete implementation of Integrand """

from ._integrand import Integrand


class QuickConstruct(Integrand):
    """ Specify and generate values of a user-defined function"""

    def __init__(self, measure, custom_fun):
        """
        Initialize custom Integrand

        Args:
            measure (TrueMeasure): a TrueMeasure instance
            custom_fun (function): a function evaluating samples (nxd) -> (nx1). See g method.
        """
        self.measure = measure
        self.custom_fun = custom_fun
        super().__init__()

    def g(self, x):
        """
        Original integrand to be integrated

        Args:
            x: nodes, $\\boldsymbol{x}_{\\mathfrak{u},i} = i^{\\mathtt{th}}$ \
               row of an $n \\cdot \\lvert \\mathfrak{u} \\rvert$ matrix

        Returns:
            $n \\cdot p$ matrix with values \
            $f(\\boldsymbol{x}_{\\mathfrak{u},i},\\mathbf{c})$ where if \
            $\\boldsymbol{x}_i' = (x_{i, \\mathfrak{u}},\\mathbf{c})_j$, \
            then $x'_{ij} = x_{ij}$ for $j \\in \\mathfrak{u}$, \
            and $x'_{ij} = c$ otherwise
        """
        return self.custom_fun(x)
