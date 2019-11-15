""" Definition for abstract class, ``Integrand`` """

from .._util import univ_repr, TransformError

from abc import ABC, abstractmethod


class Integrand(ABC):
    """
    Specify and generate values :math:`f(\\boldsymbol{x})` for \
    :math:`\\boldsymbol{x} \\in \\mathcal{X}`
    """

    def __init__(self):
        """
        Attributes:
            f (Integrand): function transformed to accept distribution \
                values
            dimension (int): Dimension of the domain, :math:`d > 0`. Default to 2.
            integrand_list (list): List of Integrands, may be more than 1 for \
                multi-dimensional problems
        """
        super().__init__()
        self.integrand_list = [self]

    @abstractmethod
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

    def f(self, x):
        """
        Transformed integrand to be integrated

        Args:
            x: nodes, :math:`\\boldsymbol{x}_{\\mathfrak{u},i} = i^{\\mathtt{th}}` \
                row of an :math:`n \\cdot |\\mathfrak{u}|` matrix

        Returns:
            :math:`n \\cdot p` matrix with values \
            :math:`f(\\boldsymbol{x}_{\\mathfrak{u},i},\\mathbf{c})` where if \
            :math:`\\boldsymbol{x}_i' = (x_{i, \\mathfrak{u}},\\mathbf{c})_j`, \
            then :math:`x'_{ij} = x_{ij}` for :math:`j \\in \\mathfrak{u}`, \
            and :math:`x'_{ij} = c` otherwise

        Note:
            To initilize this method for each integrand, call ::

                true_measure_obj.set_f(discrete_distrib_obj, integrand_obj)

            This method is not to be called directly on the original
            constructing object.
            Calls to this method should be from the i^th integrand.
            Example call: ::

                integrand_obj[i].f(x)

        Raises:
            IntegrandError if this method is called on the original \
            construcing TrueMeasure object or has not \
            been initialized for each integrand yet
        """
        raise TransformError("""
            To initilize this method for each integrand call:
                true_measure_obj.set_f(discrete_distrib_obj, integrand_obj)
            To call this method for the ith integrand call:
                integrand_obj[i].f(x)""")

    def __len__(self):
        return len(self.integrand_list)

    def __iter__(self):
        for fun in self.integrand_list:
            yield fun

    def __getitem__(self, i):
        return self.integrand_list[i]

    def __setitem__(self, i, val):
        self.integrand_list[i] = val

    def __repr__(self, attributes=[]):
        """
        Print important attribute values

        Args: 
            attributes (list): list of attributes to print

        Returns:
            string of self info
        """
        return univ_repr(self[0], "Integrand", set(attributes))
