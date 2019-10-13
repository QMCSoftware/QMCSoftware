""" Definitions for concrete Measure Implementations """

from numpy import arange,cumsum,diff,insert,sqrt
from scipy.stats import norm 

from . import TrueDistribution

class Uniform(TrueDistribution):
    """ Standard Uniform Measure """
    def __init__(self, dimension, lower_bound=0, upper_bound=1):
        transforms = {
            'StdUniform': lambda self,samples: samples*(self.b-self.a) + self.a,
            'StdGaussian':lambda self,samples: norm.cdf(samples)*(self.b-self.a) + self.a}
        super().__init__(transforms, dimension, a=lower_bound, b=upper_bound)  
        
class Gaussian(TrueDistribution):
    """ Standard Gaussian Measure """
    def __init__(self, dimension, mean=0, variance=1):
        transforms = {
            'StdGaussian': lambda self,samples: self.mu + self.sigma*samples,
            'StdUniform': lambda self,samples: norm.ppf(samples, loc=self.mu, scale=self.sigma)}
        super().__init__(dimension, transforms, mu=mean, sigma=sqrt(variance))
  
class BrownianMotion(TrueDistribution):
    """ Brownian Motion Measure """
    def __init__(self, dimension, time_vector=arange(1/4,5/4,1/4)):
        transforms = {
            'StdGaussian': lambda self,samples: cumsum(samples * sqrt(diff(insert(self.time_vector, 0, 0))), 2),
            'StdUniform': lambda self,samples: cumsum(norm.ppf(samples) * sqrt(diff(insert(self.time_vector, 0, 0))), 2)}
        super().__init__(dimension, transforms, time_vector=time_vector)