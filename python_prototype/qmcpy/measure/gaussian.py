""" Definition of Gaussian, a concrete implementation of Measure """

from ._measure import Measure

from numpy import sqrt
from scipy.stats import norm


class Gaussian(Measure):
    """ Gaussian (Normal) Measure """

    def __init__(self, distribution, mean=0, variance=1):
        """
        Args:
            distribution (Distribution): Distribution instance
            mean (float): mu for Normal(mu,sigma^2)
            variance (float): sigma^2 for Normal(mu,sigma^2)
        """
        self.distrib_obj = distribution
        self.mean = mean
        self.variance = variance
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
            # shift and stretch
            tf_samples = self.mean + self.variance * samples
        elif self.distrib_obj.mimics == "StdUniform":
            # inverse CDF then shift and stretch
            tf_samples = norm.ppf(samples, loc=self.mean, scale=self.variance)
        else:
            raise TransformError(\
                'Cannot transform samples mimicing %s to Gaussian'%self.distrib_obj.mimics)
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
                'Cannot transform samples mimicing %s to Gaussian'%self.distrib_obj.mimics)
        return f

    def __repr__(self, attributes=[]):
        """
        Print important attribute values

        Args:
            attributes (list): list of attributes to print

        Returns:
            string of self info
        """
        return super().__repr__(['mean', 'variance'])
