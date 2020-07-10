from ._discrete_distribution import DiscreteDistribution
from .qrng import korobov_qrng
from numpy import random


class Korobov(DiscreteDistribution):
    """
    Quasi-Random Korobov nets.
    
    >>> k = Korobov(1,seed=7)
    >>> k.gen_samples(2)
    array([[0.982],
           [0.482]])
    >>> k.gen_samples(2)
    array([[0.982],
           [0.482]])
    >>> k.set_dimension(3)
    >>> k.set_seed(8)
    >>> k.gen_samples(4,generator=[2,3,1])
    array([[0.265, 0.153, 0.115],
           [0.765, 0.903, 0.365],
           [0.265, 0.653, 0.615],
           [0.765, 0.403, 0.865]])
    >>> k
    Korobov (DiscreteDistribution Object)
        dimension       3
        generator       1
        randomize       1
        seed            2^(3)
        mimics          StdUniform

    References
        Marius Hofert and Christiane Lemieux (2019). 
        qrng: (Randomized) Quasi-Random Number Generators. 
        R package version 0.0-7.
        https://CRAN.R-project.org/package=qrng.
    """

    parameters = ['dimension','generator','randomize','seed','mimics']

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
        self.mimics = 'StdUniform'
        self.seed = seed
        self.set_seed(self.seed)
        super(Korobov,self).__init__()

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
        """ See abstract method. """
        self.seed = seed if seed else random.randint(2**32)
        
    def set_dimension(self, dimension):
        """ See abstract method. """
        self.dimension = dimension