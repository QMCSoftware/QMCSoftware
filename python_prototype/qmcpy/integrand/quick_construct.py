""" Definition for class QuickConstruct, a concrete implementation of Integrand """

import inspect
import numpy as np
from . import Integrand


class QuickConstruct(Integrand):
    """ Specify and generate values of a user-defined function"""

    def __init__(self, custom_fun=None, dimension=2):
        """
        Initialize custom Integrand

        Args:
            custom_fun (int): a callable univariable or multivariate Python \
             function that returns a real number.
            dimension (int): Dimension of the domain, :math:`d > 0`. Default to 2.

        Note:
            Input of the function:

            x: nodes, :math:`\\boldsymbol{x}_{\\mathfrak{u},i} = i^{\\mathtt{th}}` \
                        row of an :math:`n \\cdot |\\mathfrak{u}|` matrix
        """
        super().__init__(dimension)
        if callable(custom_fun):
            self.custom_fun = custom_fun
            self.fun_str = inspect.getsource(custom_fun).strip()
        else:
            raise Exception("Input custom_fun should be a callable function.")

        try:
            if isinstance(self.dimension, int):
                x = np.random.rand(2, self.dimension)  # 2 sampling points
                y = self.g(x)
        except:
            raise Exception(
                "Input custom_fun should be able to process a numpy 2D array "
                "with each row being a sampling point in the integral domain.")

        if (not isinstance(y, np.ndarray)) or y.shape[0] != x.shape[0]:
            raise Exception("Input custom_fun should be able to return a real "
                            " value for each sampling point in input x.")

    def g(self, x):
        if (not self.dimension) or (self.dimension != x.shape[1]):
            self.dimension = x.shape[1]  # infer domain dimension
        return self.custom_fun(x)

    def summarize(self):
        """Print important attribute values
        """
        header_fmt = "%s (%s)"
        item_i = "\n%25s: %d"
        item_s = "\n%25s: %-15s"
        attrs_vals_str = header_fmt % (type(self).__name__, "Integrand Object")
        if isinstance(self.dimension, int):
            attrs_vals_str += item_i % ("dimension", self.dimension)
        attrs_vals_str += item_s % ("custom_fun", self.fun_str)
        print(attrs_vals_str)
