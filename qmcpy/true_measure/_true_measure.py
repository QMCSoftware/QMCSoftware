from ..util import MethodImplementationError, _univ_repr, DimensionError, ParameterError


class TrueMeasure(object):
    """ True Measure abstract class. DO NOT INSTANTIATE. """

    def __init__(self):
        prefix = 'A concrete implementation of TrueMeasure must have '
        if not hasattr(self, 'distribution'):
            raise ParameterError(prefix + 'self.distribution (a DiscreteDistribution instance)')
        if not hasattr(self,'parameters'):
            self.parameters = []
        self.dimension = self.distribution.dimension

    def gen_samples(self, *args, **kwargs):
        """
        ABSTRACT METHOD to generate samples from the DiscreteDistribution object
        at which to evaluate the original integrand. 
        
        Args:
            *args (tuple): Ordered arguments to self.distribution.gen_samples
            **kwrags (dict): Keyword arguments to self.distribution.gen_samples
        
        Returns:
            ndarray: samples from the DiscreteDistribution transformed to mimic the TrueMeasure.
        
        Note:
            May not be applicable for all measures (ex: Lebesgue). 
        """
        raise MethodImplementationError(self,'gen_samples')
    
    def _eval_f(self, x, g, *args, **kwargs):
        """
        ABSTRACT METHOD to evaluate the transformed integrand, f 
        
        Args:
            x (ndarray): n x d array of samples from a discrete distribution
            g (method): original integrand (Integrand.g)
            *args: ordered args to g
            **kwargs (dict): keyword args to g
        
        Returns:
            ndarray: length n vector of funciton evaluations
        """
        raise MethodImplementationError(self,'_eval_f')

    def pdf(self, x):
        """
        Probability density function

        Args:
            x (ndarray): d (dimension) vector sample at which to evaluate the pdf
        
        Note:
            May not be applicable for all measures (ex: Lebesgue).
        """ 
        raise MethodImplementationError(self,'pdf')

    def set_dimension(self, dimension):
        """
        ABSTRACT METHOD to reset the dimension of the problem. 
        A wrapper around DiscreteDistribution.set_dimension.

        Args:
            dimension (int): new dimension to reset to 
        
        Note:
            May not be applicable for all measures (ex: Gaussian with covariance != k*eye(d) for scalar k)
        """
        raise DimensionError("Cannot reset dimension of %s object"%str(type(self).__name__))

    def __repr__(self):
        return _univ_repr(self, "TrueMeasure", self.parameters)
    
    def plot(self, *args, **kwargs):
        """ Create a plot relevant to the true measure object. """
        raise MethodImplementationError(self,'plot')
