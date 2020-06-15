from ._discrete_distribution import DiscreteDistribution
from .qrng import korobov_qrng
from numpy import random


class Korobov(DiscreteDistribution):
    """
    Quasi-Random Korobov nets.
    
    References
        Marius Hofert and Christiane Lemieux (2019). 
        qrng: (Randomized) Quasi-Random Number Generators. 
        R package version 0.0-7.
        https://CRAN.R-project.org/package=qrng.
    """

    parameters = ['dimension','generator','randomize','seed']

    def __init__(self, dimension=1, generator=[1], randomize=True, seed=None):
        """
        Args:
            dimension (int): dimension of samples
            generator (ndarray of ints): generator in {1,..,n-1}
                either a vector of length d
                or a single number (which is appropriately extended)
            randomize (bool): randomize the Korobov sequence? 
            seed (int): seed the random number generator for reproducibility
        """
        self.dimension = dimension
        self.generator = generator
        self.randomize = randomize
        self.seed = seed
        self.set_seed(self.seed)

    def gen_samples(self, n=8, generator=None):
        """
        Generate samples

        Args:
            n (int): number of samples
            generator (ndarray of ints): generator in {1,..,n-1}
                either a vector of length d
                or a single number (which is appropriately extended)

        Returns:
            ndarray: n x d (dimension) array of samples
        """
        if not generator:
            generator = self.generator
        x = korobov_qrng(n, self.dimension, generator, self.randomize, self.seed)
        return x

    def set_seed(self, seed):
        """
        Reseed the generator to get a new scrambling.

        Args:
            seed (int): new seed for generator
        """
        self.seed = seed if seed else random.randint(2**32)
        
    def set_dimension(self, dimension):
        """ See abstract method. """
        self.dimension = dimension