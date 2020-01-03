""" Definition of Gaussian, a concrete implementation of TrueMeasure """

from numpy import sqrt

from ._true_measure import TrueMeasure
from .._util import norm_inv_cdf_avoid_inf as inv_norm_cf


class Gaussian(TrueMeasure):
    """ Gaussian (Normal) Measure """

    def __init__(self, dimension, mean=0, variance=1):
        """
        Args:
            dimension (ndarray): dimension's' of the integrand's'
            mean (float): mu for Normal(mu,sigma^2)
            variance (float): sigma^2 for Normal(mu,sigma^2)
        """
        transforms = {
            "StdGaussian": [
                lambda self, samples: self.mu + self.sigma * samples,
                    # shift and stretch
                lambda self, g: g], # no weight
            "StdUniform": [
                lambda self, samples: inv_norm_cf(samples, loc=self.mu, scale=self.sigma),
                    # inverse CDF then shift and stretch
                lambda self, g: g] # no weight
            }
        super().__init__(dimension, [transforms],
                         mu=mean,
                         sigma=sqrt(variance))

    def __repr__(self, attributes=[]):
        """
        Print important attribute values

        Args:
            attributes (list): list of attributes to print

        Returns:
            string of self info
        """
        return super().__repr__(['mu', 'sigma'])
