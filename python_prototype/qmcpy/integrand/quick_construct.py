""" Definition for class QuickConstruct, a concrete implementation of Integrand """

from ._integrand import Integrand

import inspect
import numpy as np


class QuickConstruct(Integrand):
    """ Specify and generate values of a user-defined function"""

    def __init__(self, dimension, custom_fun):
        """
        Initialize custom Integrand

        Args:
            dimension (ndarray): dimension(s) of the integrand(s)
            custom_fun (int): a callable univariable or multivariate Python \
             function that returns a real number.

        Note:
            Input of the function:

            x: nodes, :math:`\\boldsymbol{x}_{\\mathfrak{u},i} = i^{\\mathtt{th}}` \
                        row of an :math:`n \\cdot |\\mathfrak{u}|` matrix
        """
        super().__init__(dimension,
            g = custom_fun)
        
    def g(self, x):
        return self.custom_fun(x)

    def __repr__(self, attributes=[]):
        """
        Print important attribute values

        Args: 
            attributes (list): list of attributes to print
        
        Returns:
            string of self info
        """
        return super().__repr__()
