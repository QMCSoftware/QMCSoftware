""" Definition of BrownianMotion, a concrete implementation of TrueMeasure """

from ._true_measure import TrueMeasure
from ..util import TransformError
from numpy import arange, cumsum, diff, insert, sqrt, array
from scipy.stats import norm


class BrownianMotion(TrueMeasure):
    """ Brownian Motion TrueMeasure """

    parameters = ['time_vector']

    def __init__(self, distribution, time_vector=arange(1 / 4, 5 / 4, 1 / 4)):
        """
        Args:
            distribution (DiscreteDistribution): DiscreteDistribution instance
            time_vector (list of ndarrays): monitoring times for the Integrand's'
        """
        self.distribution = distribution
        self.time_vector = array(time_vector)
        super().__init__()
    
    def _tf_to_mimic_samples(self, samples):
        """
        Transform samples to appear BrownianMotion
        
        Args:
            samples (ndarray): samples from a discrete distribution
        
        Return:
             mimic_samples (ndarray): samples from the DiscreteDistribution transformed to appear 
                                  to appear like the TrueMeasure object
        """
        if self.distribution.mimics == 'StdGaussian':
            # insert start time then cumulative sum over monitoring times
            mimic_samples = cumsum(samples * sqrt(diff(insert(self.time_vector, 0, 0))), -1)
        elif self.distribution.mimics == "StdUniform":
            # inverse CDF, insert start time, then cumulative sum over monitoring times
            mimic_samples = cumsum(norm.ppf(samples) * sqrt(diff(insert(self.time_vector, 0, 0))), -1)
        else:
            raise TransformError(\
                'Cannot transform samples mimicing %s to Brownian Motion'%self.distribution.mimics)
        return mimic_samples

    def transform_g_to_f(self, g):
        """
       Transform g, the origianl integrand, to f,
        the integrand accepting standard distribution sampels.  
        
        Args:
            g (method): original integrand
        
        Returns:
            f (method): transformed integrand
        """
        f = lambda samples: g(self._tf_to_mimic_samples(samples))
        return f
    
    def gen_mimic_samples(self, *args, **kwargs):
        """
        Generate samples from the DiscreteDistribution object
        and transform them to mimic TrueMeasure samples
        
        Args:
            *args (tuple): Ordered arguments to self.distribution.gen_samples
            **kwrags (dict): Keyword arguments to self.distribution.gen_samples
        
        Return:
             mimic_samples (ndarray): samples from the DiscreteDistribution transformed to appear 
                                  to appear like the TrueMeasure object
        """
        samples = self.distribution.gen_samples(*args,**kwargs)
        mimic_samples = self._tf_to_mimic_samples(samples)
        return mimic_samples
