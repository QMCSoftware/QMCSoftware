from ..util import MethodImplementationError, TransformError, univ_repr, ParameterError


class Integrand(object):
    """ Integrand abstract class. DO NOT INSTANTIATE. """

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
        """
        ABSTRACT METHOD for original integrand to be integrated.

        Args:
            x (ndarray): n samples by d dimension array of samples 
                generated according to the true measure. 
            l (int): OPTIONAL input for multi-level integrands. The level to generate at. 
                Note that the dimension of x is determined by the dim_at_level method for 
                multi-level methods.

        Return:
            ndarray: n vector of function evaluations
        """
        raise MethodImplementationError(self, 'g')
    
    def dim_at_level(self, l):
        """
        ABSTRACT METHOD to return the dimension of samples to generate at level l. 
        This method only needs to be implemented for multi-level integrands where 
        the dimension changes depending on the level. 
        
        Args:
            l (int): level
        
        Return:
            int: dimension of samples needed at level l
        """
        raise MethodImplementationError(self, 'dim_at_level')

    def __repr__(self):
        return univ_repr(self, "Integrand", self.parameters)
