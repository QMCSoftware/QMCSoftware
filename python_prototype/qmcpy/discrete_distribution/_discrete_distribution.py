""" DiscreteDistribution is an abstract class. """

from .._util import univ_repr, ParameterError

from abc import ABC, abstractmethod


class DiscreteDistribution(ABC):
    """
    Discrete Distribution from which we can generate samples
    
    Attributes:
        mimics (string): True Measure mimiced by the Discrete Distribution
    """

    def __init__(self):
        """ Initialize Discrete Distributuion instance """
        prefix = 'A concrete implementation of DiscreteDistribution must have ' 
        if not hasattr(self, 'mimics'):
            raise ParameterError(prefix + 'self.mimcs (measure mimiced by the distribution)')

    @abstractmethod
    def gen_dd_samples(self, replications, n_samples, dimensions):
        """
        Generate r nxd IID Standard Gaussian samples

        Args:
            replications (int): Number of nxd matrices to generate (sample.size()[0])
            n_samples (int): Number of observations (sample.size()[1])
            dimensions (int): Number of dimensions (sample.size()[2])

        Returns:
            replications x n_samples x dimensions (numpy array)
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
