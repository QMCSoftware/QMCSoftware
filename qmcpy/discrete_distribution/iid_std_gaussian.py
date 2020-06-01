from ._discrete_distribution import DiscreteDistribution
from numpy.random import Generator, PCG64


class IIDStdGaussian(DiscreteDistribution):

    parameters = ['dimension', 'seed', 'mimics']

    def __init__(self, dimension=1, seed=None):
        """
        Args:
            dimension (int): dimension of samples
            seed (int): seed the random number generator for reproducibility
        """
        self.dimension = dimension
        self.seed = seed
        self.rng = Generator(PCG64(self.seed))
        self.mimics = 'StdGaussian'
        super().__init__()

    def gen_samples(self, n):
        """
        Generate samples 

        Args:
            n (int): Number of observations to generate

        Returns:
            ndarray: n x d (dimension) array of samples
        """
        return self.rng.standard_normal((int(n), self.dimension))
    
    def set_dimension(self, dimension):
        """ See abstract method. """
        self.dimension = dimension
