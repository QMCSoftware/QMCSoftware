""" Definition of BrownianMotion, a concrete implementation of Measure """

from ._measure import Measure
from ..uitl import TransformError
from numpy import arange, cumsum, diff, insert, sqrt
from scipy.stats import norm


class BrownianMotion(Measure):
    """ Brownian Motion Measure """

    def __init__(self, distribution, time_vector=arange(1 / 4, 5 / 4, 1 / 4)):
        """
        Args:
            distribution (Distribution): Distribution instance
            time_vector (list of ndarrays): monitoring times for the Integrand's'
        """
        self.distrib_obj = distribution
        self.time_vector = time_vector
        super().__init__()
    
    def gen_samples(self, *args, **kwargs)):
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
        samples = self.distrib_obj.gen_samples(*args,**kwargs)
        if self.distrib_obj.mimics == 'StdGaussian':
            # insert start time then cumulative sum over monitoring times
            tf_samples = cumsum(samples * sqrt(diff(insert(self.time_vector, 0, 0))), 2)
        elif self.distrib_obj.mimics == "StdUniform":
            # inverse CDF, insert start time, then cumulative sum over monitoring times
            tf_samples = cumsum(norm.ppf(samples) sqrt(diff(insert(self.time_vector, 0, 0))), 2)
        else:
            raise TransformError(\
                'Cannot transform samples mimicing %s to Brownian Motion'%self.distrib_obj.mimics)
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
        if self.distrib_obj.mimics in ['StdUniform','StdGaussian']:
            # no weight
            f = lambda tf_samples: g(tf_samples)
        else:
            raise TransformError(\
                'Cannot transform samples mimicing %s to Brownian Motion'%self.distrib_obj.mimics)
        return f

    def __repr__(self, attributes=[]):
        """
        Print important attribute values

        Args:
            attributes (list): list of attributes to print

        Returns:
            string of self info
        """
        return super().__repr__(['time_vector'])
