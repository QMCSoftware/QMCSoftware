""" Definition of Lebesgue, a concrete implementation of Measure """

from ._measure import Measure
from ..util import TransformError
from numpy import array


class Lebesgue(Measure):
    """ Lebesgue Uniform Measure """

    def __init__(self, distribution, lower_bound=0., upper_bound=1):
        """
        Args:
            distribution (Distribution): Distribution instance
            lower_bound (float): a for Uniform(a,b)
            upper_bound (float): b for Uniform(a,b)
        """
        self.distribution = distribution
        self.lower_bound = array(lower_bound)
        self.upper_bound = array(upper_bound)
        super().__init__()

    def gen_samples(self, *args, **kwargs):
        """
        Generate samples from the Distribution object
        and transform them to mimic Measure samples
        
        Args:
            *args (tuple): Ordered arguments to self.distribution.gen_samples
            **kwrags (dict): Keyword arguments to self.distribution.gen_samples
        
        Returns:
            tf_samples (ndarray): samples from the Distribution object transformed to appear 
                                  to appear like the Measure object
        """
        samples = self.distribution.gen_samples(*args,**kwargs)
        if self.distribution.mimics == "StdUniform":
            # stretch samples
            tf_samples = samples * (self.upper_bound - self.lower_bound) + self.lower_bound
        else:
            raise TransformError(\
                'Cannot transform samples mimicing %s to Lebesgue'%self.distribution.mimics)
        return tf_samples

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
        if self.distribution.mimics in ['StdUniform']:
            # multiply dimensional difference
            f = lambda tf_samples: g(tf_samples) * (self.upper_bound - self.lower_bound).prod()
        else:
            raise TransformError(\
                'Cannot transform samples mimicing %s to Lebesgue'%self.distribution.mimics)
        return f

    def __repr__(self, attributes=[]):
        """
        Print important attribute values

        Args: 
            attributes (list): list of attributes to print

        Returns:
            string of self info
        """
        return super().__repr__(['lower_bound', 'upper_bound'])
