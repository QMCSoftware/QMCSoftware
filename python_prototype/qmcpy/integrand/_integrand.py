""" Definition for abstract class, ``Integrand`` """

from ..util import MethodImplementationError, TransformError, univ_repr, ParameterError


class Integrand(object):
    """
    Specify and generate values :math:`f(\\boldsymbol{x})` for \
    :math:`\\boldsymbol{x} \\in \\mathcal{X}`
    """

    def __init__(self):
        prefix = 'A concrete implementation of Integrand must have '
        if not hasattr(self, 'measure'):
            raise ParameterError(prefix + 'self.measure (a Measure instance)')
        if not hasattr(self,'parameters'):
            self.parameters = []
        self.dimension = self.measure.dimension
        self.f = self.measure.transform_g_to_f(self.g) # transformed integrand
        
    def g(self, x):
        """ ABSTRACT METHOD
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
        raise MethodImplementationError(self, 'g')

    def __repr__(self):
        return univ_repr(self, "Integrand", self.parameters)
