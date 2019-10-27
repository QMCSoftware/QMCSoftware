""" Definition for class QuickConstruct, a concrete implementation of Integrand """

from . import Integrand


class QuickConstruct(Integrand):
    """ Specify and generate values of a user-defined function"""

    def __init__(self, custom_fun=None, dimension=2):
        """
        Initialize custom Integrand

        Args:
            custom_fun (int): a callable univariable or multivariate Python \
             function that returns a real number.

        Note:
            Input of the function:

            x: nodes, :math:`\\boldsymbol{x}_{\\mathfrak{u},i} = i^{\\mathtt{th}}` \
                        row of an :math:`n \\cdot |\\mathfrak{u}|` matrix
        """
        super().__init__(dimension)
        if callable(custom_fun):
            self.custom_fun = custom_fun
        else:
            raise Exception("Input custom_fun should be a callable function.")

    def g(self, x):
        return self.custom_fun(x)
