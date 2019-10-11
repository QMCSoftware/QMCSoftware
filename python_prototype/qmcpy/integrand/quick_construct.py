""" Definition for class QuickConstruct, a concrete implementation of Integrand """

from . import Integrand


class QuickConstruct(Integrand):
    """ Specify and generate values of custom user-function"""


    def __init__(self, nominal_value=None, custom_fun=None):
        """
        Initialize custom Integrand

        Args:
            nominal_value (int): :math:`c` such that\
                :math:`(c, \ldots, c) \in \mathcal{X}`
            custom_fun (int): function pointer.
                input to pointed to function:
                    x: nodes, :math:`\mathbf{x}_{\mathfrak{u},i} = i^{\mathtt{th}}` \
                        row of an :math:`n \cdot |\mathfrak{u}|` matrix
                    coord_index: set of those coordinates in sequence needed, \
                        :math:`\mathfrak{u}`  
        """
        super().__init__(nominal_value=nominal_value)
        self.custom_fun = custom_fun

    def g(self, x, coord_index):
        return self.custom_fun(x, coord_index)

