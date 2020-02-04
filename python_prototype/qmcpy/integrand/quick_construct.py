""" Definition for class QuickConstruct, a concrete implementation of Integrand """

from ._integrand import Integrand


class QuickConstruct(Integrand):
    """ Specify and generate values of a user-defined function"""

    def __init__(self, measure, custom_fun):
        """
        Initialize custom Integrand

        Args:
            measure (Measure): a Measure instance
            custom_fun (function): a function evaluating samples (nxd) -> (nx1)
                Args:
                    x: nodes, :math:`\\boldsymbol{x}_{\\mathfrak{u},i} = i^{\\mathtt{th}}` \
                       row of an :math:`n \\cdot |\\mathfrak{u}|` matrix
        """
        self.measure = measure
        self.custom_fun = custom_fun
        super().__init__()

    def g(self, x):
        return self.custom_fun(x)
