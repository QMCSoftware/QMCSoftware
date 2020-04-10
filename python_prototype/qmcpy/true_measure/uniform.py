""" Definition of Uniform, a concrete implementation of TrueMeasure """

from ._true_measure import TrueMeasure
from ..util import TransformError
from numpy import array, prod
from scipy.stats import norm


class Uniform(TrueMeasure):
    """ Uniform TrueMeasure """

    parameters = ['lower_bound', 'upper_bound']
    
    def __init__(self, distribution, lower_bound=0., upper_bound=1.):
        """
        Args:
            distribution (DiscreteDistribution): DiscreteDistribution instance
            lower_bound (float): a for Uniform(a,b)
            upper_bound (float): b for Uniform(a,b)
        """
        self.distribution = distribution
        self.lower_bound = array(lower_bound)
        self.upper_bound = array(upper_bound)
        super().__init__()
    
    def pdf(self,x):
        return 1/( prod(self.upper_bound-self.lower_bound) )

    def _tf_to_mimic_samples(self, samples):
        """
        Transform samples to appear Uniform
        
        Args:
            samples (ndarray): samples from a discrete distribution
        
        Return:
             mimic_samples (ndarray): samples from the DiscreteDistribution transformed to appear 
                                  to appear like the TrueMeasure object
        """
        if self.distribution.mimics == 'StdGaussian':
            # CDF then stretch
            mimic_samples = norm.cdf(samples) * (self.upper_bound - self.lower_bound) + self.lower_bound
        elif self.distribution.mimics == "StdUniform":
            # stretch samples
            mimic_samples = samples * (self.upper_bound - self.lower_bound) + self.lower_bound
        else:
            raise TransformError(\
                'Cannot transform samples mimicing %s to Uniform'%self.distribution.mimics)
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
        
        Returns:
            mimic_samples (ndarray): samples from the DiscreteDistribution object transformed to appear 
                                  to appear like the TrueMeasure object
        """
        samples = self.distribution.gen_samples(*args,**kwargs)
        mimic_samples = self._tf_to_mimic_samples(samples)
        return mimic_samples
