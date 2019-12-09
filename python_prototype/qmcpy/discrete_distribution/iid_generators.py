""" This module implements mutiple subclasses of DiscreteDistribution. """

from ._discrete_distribution import DiscreteDistribution

from numpy import array
from numpy.random import Generator, PCG64

class IIDStdUniform(DiscreteDistribution):
    """ IID Standard Uniform """

    def __init__(self, rng_seed=None):
        """
        Args:
            rng_seed (int): seed the random number generator for reproducibility
        """
        self.mimics = 'StdUniform'
        self.rng = Generator(PCG64(rng_seed))
        super().__init__()

    def gen_dd_samples(self, replications, n_samples, dimensions):
        """
        Generate r nxd IID Standard Uniform samples

        Args:
            replications (int): Number of nxd matrices to generate (sample.size()[0])
            n_samples (int): Number of observations (sample.size()[1])
            dimensions (int): Number of dimensions (sample.size()[2])

        Returns:
            replications x n_samples x dimensions (numpy array)
        """
        r = int(replications)
        n = int(n_samples)
        d = int(dimensions)
        return self.rng.uniform(0, 1, (r, n, d))


class IIDStdGaussian(DiscreteDistribution):
    """ Standard Gaussian """

    def __init__(self, rng_seed=None):
        """
        Args:
            rng_seed (int): seed the random number generator for reproducibility
        """
        self.mimics = 'StdGaussian'
        self.rng = Generator(PCG64(rng_seed))
        super().__init__()

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
        r = int(replications)
        n = int(n_samples)
        d = int(dimensions)
        return self.rng.standard_normal((r, n, d))