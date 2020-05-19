""" Definition for abstract class, ``Integrand`` """

from ..util import MethodImplementationError, TransformError, univ_repr, ParameterError


class Integrand(object):
    """
    Specify and generate values $f(\\boldsymbol{x})$ for \
    $\\boldsymbol{x} \\in \\mathcal{X}$
    """

    def __init__(self):
        prefix = 'A concrete implementation of Integrand must have '
        if not hasattr(self, 'measure'):
            raise ParameterError(prefix + 'self.measure (a TrueMeasure instance)')
        if not hasattr(self,'parameters'):
            self.parameters = []
        self.dimension = self.measure.dimension
        self.f = self.measure.transform_g_to_f(self.g) # transformed integrand
        if not hasattr(self,'multilevel'):
            self.multilevel = False
        
    def g(self, x):
        """ ABSTRACT METHOD
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
        raise MethodImplementationError(self, 'g')
    
    def dim_at_level(self, l):
        """ ABSTRACT METHOD
        Return the dimension of samples to generate for level l.
        Will call Measure.set_dimension on returned level to get samples for new level
        """
        raise TransformError("Integrand does not have dim_at_level method")

    def __repr__(self):
        return univ_repr(self, "Integrand", self.parameters)
