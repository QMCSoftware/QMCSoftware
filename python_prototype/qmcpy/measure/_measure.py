""" Abstract Class Measure """

from copy import deepcopy

from ..util import multilevel_constructor, MethodImplementationError, univ_repr


class Measure(object):
    """ The True Measure of the Integrand """

    def __init__(self):
        prefix = 'A concrete implementation of Measure must have '
        if not hasattr(self, 'distrib_obj'):
            raise ParameterError(prefix + 'self.distrib_obj (a Distribution instance)')
        self.distribution = type(self.distrib_obj).__name__

    def gen_samples(self, *args, **kwargs):
        """
        Generate samples from the Distribution object
        and transform them to mimic Measure samples
        
        Args:
            *args (tuple): Ordered arguments to self.distrib_obj.gen_samples
            **kwrags (dict): Keyword arguments to self.distrib_obj.gen_samples
        
        Returns:
            tf_samples (ndarray): samples from the Distribution object transformed
                                  to appear like the Measure object
        """
        raise MethodImplementationError(self,'gen_samples')

    def transform_g_to_f(self, g):
        """
        Transform the g, the origianl integrand, to f,
        the integrand after transforming Distribution samples
        to mimic the Measure object. 
        
        Args:
            g (method): original integrand
        
        Returns:
            f (method): transformed integrand
        """
        raise MethodImplementationError(self,'transform_g_to_f')

    def __repr__(self, attributes=[]):
        """
        Print important attribute values

        Args:
            attributes (list): list of attributes to print

        Returns:
            string of self info
        """
        return univ_repr(self, "True Measure", attributes)
