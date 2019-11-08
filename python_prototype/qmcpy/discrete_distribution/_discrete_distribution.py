""" DiscreteDistribution is an abstract class. """

from abc import ABC, abstractmethod

from .._util import univ_repr


class DiscreteDistribution(ABC):
    """ Discrete Distribution from which we can generate samples. """

    def __init__(self, mimics, rng_seed):
        """
        Initialize Discrete Distributuion instance

        Args:
            mimics (str): Measure the discrete distribution attempts to mimic
            rng_seed (int): seed for whatever generator is to be used
        """
        self.mimics = mimics
        self.rng_seed = rng_seed

    @abstractmethod
    def gen_dd_samples(self, r, n, d):
        """
        Generate r nxd IID Standard Gaussian samples

        Args:
            r (int): Number of nxd matrices to generate (sample.size()[0])
            n (int): Number of observations (sample.size()[1])
            d (int): Number of dimensions (sample.size()[2])

        Returns:
            rxnxd (numpy array)
        """
        return

    def __repr__(self, attributes=[]):
        """
        Print important attribute values

        Args: 
            attributes (list): list of attributes to print

        Returns:
            string of self info
        """
        attributes = set(attributes + ['mimics', 'rng_seed'])
        return univ_repr(self, "Discrete Distribution", attributes)
