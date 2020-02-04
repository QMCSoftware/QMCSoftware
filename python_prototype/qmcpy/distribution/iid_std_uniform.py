""" Definition for IIDStdUniform, a concrete implementation of Distribution """

from ._distribution import Distribution
from numpy.random import Generator, PCG64


class IIDStdUniform(Distribution):
    """ Standard Uniform """

    parameters = ['dimension','seed','mimics']

    def __init__(self, dimension=1, seed=None):
        """
        Args:
            dimension (int): dimension of samples
            seed (int): seed the random number generator for reproducibility
        """
        self.dimension = dimension
        self.seed = seed
        self.rng = Generator(PCG64(self.seed))
        self.mimics = 'StdUniform'
        super().__init__()

    def gen_samples(self, n):
        """
        Generate n x self.dimension IID Standard Uniform samples

        Args:
            n (int): Number of observations to generate

        Returns:
            n x self.dimension (ndarray)
        """
        return self.rng.uniform(0,1,(n, self.dimension))
        