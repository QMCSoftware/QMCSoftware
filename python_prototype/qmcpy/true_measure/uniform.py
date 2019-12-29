""" Definition of Uniform, a concrete implementation of TrueMeasure """

from ._true_measure import TrueMeasure

from scipy.stats import norm


class Uniform(TrueMeasure):
    """ Uniform Measure """

    def __init__(self, dimension, lower_bound=0., upper_bound=1.):
        """
        Args:
            dimension (ndarray): dimension's' of the integrand's'
            lower_bound (float): a for Uniform(a,b)
            upper_bound (float): b for Uniform(a,b)
        """
        transforms = {
            "StdUniform": [
                lambda self, samples: samples * (self.b - self.a) + self.a,
                    # stretch samples
                lambda self, g: g], # no weight
            "StdGaussian": [
                lambda self, samples: norm.cdf(samples) * (self.b - self.a) + self.a,
                    # CDF then stretch
                lambda self, g: g] # no weight
            }
        super().__init__(dimension, [transforms],
                         a=lower_bound,
                         b=upper_bound)

    def __repr__(self, attributes=[]):
        """
        Print important attribute values

        Args: 
            attributes (list): list of attributes to print

        Returns:
            string of self info
        """
        return super().__repr__(['a', 'b'])
