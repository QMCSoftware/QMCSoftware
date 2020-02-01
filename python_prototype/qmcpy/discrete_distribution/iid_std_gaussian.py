""" Definition for IIDStdGaussian, a concrete implementation of DiscreteDistribution """

from numpy.random import Generator, PCG64

from ._discrete_distribution import DiscreteDistribution


class IIDStdGaussian(DiscreteDistribution):
    """ Standard Gaussian """

    def __init__(self, dimension=1, replications=0, seed=None):
        """
        Args:
            dimension (int): dimension of samples
            replications (int): Number of nxd matrices to generate
                replications set to 0 ignores replications and returns (n_max-n_min)xd samples            
            seed (int): seed the random number generator for reproducibility
        """
        self.dimension = dimension
        self.squeeze = (replications==0)
        self.replications = max(replications,1)
        self.mimics = 'StdGaussian'
        self.seed = seed
        self.rng = Generator(PCG64(self.seed))
        super().__init__()

    def gen_samples(self, n):
        """
        Generate self.replications nxself.dimension IID Standard Gaussian samples

        Args:
            n (int): Number of observations to generate

        Returns:
            self.replications x n x self.dimension (ndarray)
        """
        x = self.rng.standard_normal((self.replications, n, self.dimension))
        if self.squeeze:
            x = x.squeeze(0)
        return x