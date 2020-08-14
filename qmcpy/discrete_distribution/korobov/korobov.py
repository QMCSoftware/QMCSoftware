from .._discrete_distribution import DiscreteDistribution
from ...util import ParameterError
from .korobov_qrng import KorobovQRNG
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
        backend         QRNG
    
    References:

        [1] Marius Hofert and Christiane Lemieux (2019). 
        qrng: (Randomized) Quasi-Random Number Generators. 
        R package version 0.0-7.
        https://CRAN.R-project.org/package=qrng.
    """

    parameters = ['dimension','generator','randomize','seed','mimics','backend']

    def __init__(self, dimension=1, generator=[1], randomize=True, seed=None, backend='QRNG'):
        """
        Args:
            dimension (int): dimension of samples
            generator (ndarray of ints): generator in {1,..,n-1}
                either a vector of length d
                or a single number (which is appropriately extended)
            randomize (bool): randomize the Korobov sequence? 
                Note: Non-randomized Korobov sequence includes origin
            seed (int): seed the random number generator for reproducibility
            backend (str): backend generator must be "QRNG"
        """
        self.backend = backend.upper()
        backend_objs = {'QRNG':KorobovQRNG}
        backends = list(backend_objs.keys())
        if self.backend not in backends:
            raise ParameterError('Korobov requires backend be in %s'%(str(backends)))
        self.dimension = dimension
        self.generator_obj = backend_objs[self.backend](dimension,generator,randomize,seed)
        self.low_discrepancy = True
        self.mimics = 'StdUniform'
        super(Korobov,self).__init__()

    def gen_samples(self, n=None, n_min=0, n_max=8, warn=True):
        """
        Generate samples

        Args:
            n (int): number of samples

        Returns:
            ndarray: n x d (dimension) array of samples
        """
        if n:
            n_max = n
            n_min = 0
        x = self.generator_obj.gen_samples(n_min,n_max,warn)
        return x

    def set_seed(self, seed):
        """ See abstract method. """
        self.generator_obj.set_seed(seed)
        
    def set_dimension(self, dimension):
        """ See abstract method. """
        self.generator_obj.set_dimension(dimension)
    
    def __repr__(self):
        self.dimension, self.generator, self.randomize, self.seed = self.generator_obj.get_params()
        return super().__repr__()
