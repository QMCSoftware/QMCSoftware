""" Definitions of TrueMeasure Concrete Classes """

from numpy import arange, cumsum, diff, insert, sqrt
from scipy.stats import norm

from ._true_measure import TrueMeasure


class Uniform(TrueMeasure):
    """ Uniform Measure """

    def __init__(self, dimension, lower_bound=0., upper_bound=1.):
        """
        Args:
            dimension (ndarray): dimension's' of the integrand's'
            lower_bound (float): a for Uniform(a,b)
            upper_bound (float): b for Uniform(a,b)
        """
        self.transforms = {
            "StdUniform": [
                lambda self, samples: samples * (self.b - self.a) + self.a,
                    # stretch samples
                lambda self, g: g], # no weight
            "StdGaussian": [
                lambda self, samples: norm.cdf(samples) * (self.b - self.a) + self.a,
                    # CDF then stretch
                lambda self, g: g] # no weight
            }
        super().__init__(dimension, a=lower_bound, b=upper_bound)
    
    def __repr__(self, attributes=[]):
        """
        Print important attribute values

        Args: 
            attributes (list): list of attributes to print
        
        Returns:
            string of self info
        """
        return super().__repr__(['a', 'b'])


class Gaussian(TrueMeasure):
    """ Gaussian (Normal) Measure """

    def __init__(self, dimension, mean=0, variance=1):
        """
        Args:
            dimension (ndarray): dimension's' of the integrand's'
            mean (float): mu for Normal(mu,sigma^2)
            variance (float): sigma^2 for Normal(mu,sigma^2)
        """
        self.transforms = {
            "StdGaussian": [
                lambda self, samples: self.mu + self.sigma * samples,
                    # shift and stretch
                lambda self, g: g], # no weight
            "StdUniform": [
                lambda self, samples: norm.ppf(samples, loc=self.mu, scale=self.sigma),
                    # inverse CDF then shift and stretch
                lambda self, g: g] # no weight
            }
        super().__init__(dimension, mu=mean, sigma=sqrt(variance))
    
    def __repr__(self, attributes=[]):
        """
        Print important attribute values

        Args: 
            attributes (list): list of attributes to print
        
        Returns:
            string of self info
        """
        return super().__repr__(['mu', 'sigma'])


class BrownianMotion(TrueMeasure):
    """ Brownian Motion Measure """

    def __init__(self, dimension, time_vector=[arange(1 / 4, 5 / 4, 1 / 4)]):
        """
        Args:
            dimension (ndarray): dimension's' of the integrand's'
            time_vector (list of ndarrays): monitoring times for the Integrand's'
        """
        self.transforms = {
            "StdGaussian": [
                lambda self, samples: cumsum(
                    samples * sqrt(diff(insert(self.time_vector, 0, 0))), 2),
                    # insert start time then cumulative sum over monitoring times
                lambda self, g: g], # no weight
            "StdUniform": [
                lambda self, samples: cumsum(norm.ppf(samples) \
                    * sqrt(diff(insert(self.time_vector, 0, 0))), 2),
                    # inverse CDF, insert start time, then cumulative sum over monitoring times
                lambda self, g: g] # no weight
            }

        super().__init__(dimension, time_vector=time_vector)
    
    def __repr__(self, attributes=[]):
        """
        Print important attribute values

        Args: 
            attributes (list): list of attributes to print
        
        Returns:
            string of self info
        """
        return super().__repr__(['time_vector'])

class Lebesgue(TrueMeasure):
    """ Lebesgue Measure """

    def __init__(self, dimension,
                 uniform_lower_bound=0., uniform_upper_bound=1.,  # for Uniform
                 gaussian_mean=0, gaussian_variance=1):  # for Gaussian
        """
        Args:
            dimension (ndarray): dimension's' of the integrand's'
        """
        self.transforms = {
            "StdUniform": [
                lambda self, samples: samples * (self.b - self.a) + self.a,
                    # stretch samples
                lambda self, g: g * (self.b - self.a).prod()]  # multiply dimensional difference
            }
        super().__init__(dimension,
                         a=uniform_lower_bound, b=uniform_upper_bound,
                         mu=gaussian_mean, sigma=sqrt(gaussian_variance))
    
    def __repr__(self, attributes=[]):
        """
        Print important attribute values

        Args: 
            attributes (list): list of attributes to print
        
        Returns:
            string of self info
        """
        return super().__repr__(['a', 'b', 'mu', 'sigma'])