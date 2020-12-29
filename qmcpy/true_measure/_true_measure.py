from ..util import MethodImplementationError, _univ_repr, DimensionError, ParameterError


class TrueMeasure(object):
    """ True Measure abstract class. DO NOT INSTANTIATE. """

    def __init__(self):
        prefix = 'A concrete implementation of TrueMeasure must have '
        if not hasattr(self,'d'):
            raise ParameterError(prefix + 'self.d')
        if not hasattr(self,'parameters'):
            self.parameters = []

    def transform(self, x): 
        """ 
        Transformation, \Psi. 

        Args:
            x: n x d matrix of samples mimicking a standard uniform.

        Returns:
            ndarray: n x d matrix of transformed x.  

        """
        raise MethodImplementationError(self,'transform')
    
    def jacobian(self, x):
        """
        ABSTRACT method to evaluate the Jacobian at sampling locations.

        Args:
            x (ndarray): n x d matrix of samples
        
        Returns:
            ndarray: length n vector of Jacobian values at locations of x
        """ 
        raise MethodImplementationError(self,'jacobian')

    def weight(self, x):
        """
        Non-negative weight function, \lambda. 
        This is often a PDF, but is not required to be 
        i.e. Lebesgue weight is always 1, but is not a PDF.

        Args:
            x (ndarray): n x d  matrix of samples
        
        Returns:
            ndarray: length n vector of weights at locations of x
        """ 
        raise MethodImplementationError(self,'weight')

    def set_dimension(self, dimension):
        """
        ABSTRACT METHOD to reset the dimension of the problem. 

        Args:
            dimension (int): new dimension to reset to 
        """
        raise DimensionError("Cannot reset dimension of %s object"%str(type(self).__name__))

    def __repr__(self):
        return _univ_repr(self, "TrueMeasure", self.parameters)
