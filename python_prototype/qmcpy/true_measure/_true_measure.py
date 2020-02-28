""" Abstract Class TrueMeasure """

from ..util import MethodImplementationError, univ_repr
from copy import deepcopy


class TrueMeasure(object):
    """ The True TrueMeasure of the Integrand """

    def __init__(self):
        prefix = 'A concrete implementation of TrueMeasure must have '
        if not hasattr(self, 'distribution'):
            raise ParameterError(prefix + 'self.distribution (a DiscreteDistribution instance)')
        if not hasattr(self,'parameters'):
            self.parameters = []
        self.dimension = self.distribution.dimension
        self.distrib_name = type(self.distribution).__name__

    def gen_mimic_samples(self, *args, **kwargs):
        """ ABSTRACT METHOD
        Generate samples from the DiscreteDistribution object
        and transform them to mimic TrueMeasure samples. 
        May not be applicable for all measures (ex: Lebesgue)
        
        Args:
            *args (tuple): Ordered arguments to self.distribution.gen_samples
            **kwrags (dict): Keyword arguments to self.distribution.gen_samples
        
        Returns:
            tf_samples (ndarray): samples from the DiscreteDistribution object transformed
                                  to appear like the TrueMeasure object
        """
        raise MethodImplementationError(self,'gen_samples')
    
    def transform_g_to_f(self, g):
        """ ABSTRACT METHOD
        Transform the g, the origianl integrand, to f,
        the integrand after transforming DiscreteDistribution samples
        to mimic the TrueMeasure object. 
        
        Args:
            g (method): original integrand
        
        Returns:
            f (method): transformed integrand
        """
        raise MethodImplementationError(self,'transform_g_to_f')

    def __repr__(self):
        return univ_repr(self, "True TrueMeasure", ['distrib_name']+self.parameters)
