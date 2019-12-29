""" Definition of BrownianMotion, a concrete implementation of TrueMeasure """

from ._true_measure import TrueMeasure
from .._util import norm_inv_cdf_avoid_inf as inv_norm_cf

from numpy import arange, cumsum, diff, insert, sqrt


class BrownianMotion(TrueMeasure):
    """ Brownian Motion Measure """

    def __init__(self, dimension, time_vector=[arange(1 / 4, 5 / 4, 1 / 4)]):
        """
        Args:
            dimension (ndarray): dimension's' of the integrand's'
            time_vector (list of ndarrays): monitoring times for the Integrand's'
        """
        transforms = {
            "StdGaussian": [
                lambda self, samples: cumsum(
                    samples * sqrt(diff(insert(self.time_vector, 0, 0))), 2),
                    # insert start time then cumulative sum over monitoring times
                lambda self, g: g], # no weight
            "StdUniform": [
                lambda self, samples: cumsum(inv_norm_cf(samples) \
                    * sqrt(diff(insert(self.time_vector, 0, 0))), 2),
                    # inverse CDF, insert start time, then cumulative sum over monitoring times
                lambda self, g: g] # no weight
            }

        super().__init__(dimension, [transforms],
                         time_vector=time_vector)

    def __repr__(self, attributes=[]):
        """
        Print important attribute values

        Args: 
            attributes (list): list of attributes to print

        Returns:
            string of self info
        """
        return super().__repr__(['time_vector'])
