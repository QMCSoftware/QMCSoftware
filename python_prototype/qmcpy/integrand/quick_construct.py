""" Definition for class QuickConstruct, a concrete implementation of Integrand """

from . import Integrand
import numpy as np
import inspect

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
            self.fun_str = inspect.getsource(custom_fun)[:-1]
        else:
            raise Exception("Input custom_fun should be a callable function.")

        try:
            x = np.array([[1, 2, 3], [4, 5, 6]])
            y = self.g(x)
        except:
            raise Exception("Input custom_fun should be able to process a numpy 2D array.")

    def g(self, x):
        return self.custom_fun(x)

    def summarize(self):
        """Print important attribute values
        """
        header_fmt = "%s (%s)"
        item_s = "\n%25s: %-15s"
        attrs_vals_str = header_fmt % (type(self).__name__, "Integrand Object")
        attrs_vals_str += item_s % ("custom_fun", self.fun_str)
        print(attrs_vals_str)