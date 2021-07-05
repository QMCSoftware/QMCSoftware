from .._discrete_distribution import DiscreteDistribution
from ...util import ParameterError,ParameterWarning
import warnings
from ..c_lib import c_lib
import ctypes
from numpy import *


class Korobov(DiscreteDistribution):
    """
    Quasi-Random Korobov nets.
    
    >>> k = Korobov(2,generator=[1,3],seed=7)
    >>> k.gen_samples(4)
    array([[0.98196076, 0.88349207],
           [0.23196076, 0.63349207],
           [0.48196076, 0.38349207],
           [0.73196076, 0.13349207]])
    >>> k.gen_samples(4)
    array([[0.98196076, 0.88349207],
           [0.23196076, 0.63349207],
           [0.48196076, 0.38349207],
           [0.73196076, 0.13349207]])
    >>> k
    Korobov (DiscreteDistribution Object)
        d               2^(1)
        generator       [1 3]
        randomize       1
        seed            7
        mimics          StdUniform
    >>> Korobov(2,generator=[3,1],seed=7).gen_samples(4)
    array([[0.98196076, 0.88349207],
           [0.73196076, 0.13349207],
           [0.48196076, 0.38349207],
           [0.23196076, 0.63349207]])
    
    References:

        [1] Marius Hofert and Christiane Lemieux (2019). 
        qrng: (Randomized) Quasi-Random Number Generators. 
        R package version 0.0-7.
        https://CRAN.R-project.org/package=qrng.
    """

    korobov_qrng_cf = c_lib.korobov_qrng
    korobov_qrng_cf.argtypes = [
        ctypes.c_int,  # n
        ctypes.c_int,  # d
        ctypeslib.ndpointer(ctypes.c_int, flags='C_CONTIGUOUS'),  # generator
        ctypes.c_int,  # randomize
        ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # result array 
        ctypes.c_uint64]  # seed
    korobov_qrng_cf.restype = None

    def __init__(self, dimension=1, generator=[1], randomize=True, seed=None):
        """
        Args:
            dimension (int): dimension of samples
            generator (ndarray of ints): generator in {1,..,n-1}
                either a vector of length d
                or a single number (which is appropriately extended)
            randomize (bool): randomize the Korobov sequence? 
                Note: Non-randomized Korobov sequence includes origin
            seed (int): seed the random number generator for reproducibility
        """
        self.parameters = ['d','generator','randomize','seed','mimics']
        
        self.generator = array(generator, dtype=int32)
        self.randomize = randomize
        self.n_lim = 2**31
        self.d_lim = self.n_lim
        self._set_dimension(dimension)
        self.set_seed(seed)
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
        l = len(self.generator)
        if l == 1:
            self.g = ((self.generator**arange(self.d, dtype=int32)) % n_max).astype(int32)
        elif l == self.d:
            self.g = self.generator
        else:
            raise ParameterError("QRNG Korobov must have generator of length 1 or dimension.")
        if (self.g<1).any() or (self.g>=n_max).any():
            raise ParameterError('QRNG Korobov requires all(1 <= generator ints <= (n-1)).')
        if self.randomize==False and warn:
            warnings.warn("Non-randomized Korobov sequence includes the origin.",ParameterWarning)
        if n_min>0:
            raise ParameterError('QRNG Korobov does not support skipping samples with n_min>0.')
        if n_max < 2:
            raise ParameterError('QRNG Korobov requires n>=2.')
        if n_max > self.n_lim:
            raise Exception('QRNG Korobov requires n must be <=2^32')
        x = zeros((self.d, n_max), dtype=double)
        self.korobov_qrng_cf(int(n_max), int(self.d), self.g, self.randomize, x, self.seed)
        return x.T

    def pdf(self, x):
        """ pdf of a standard uniform """
        return ones(x.shape[0], dtype=float)
        
    def _set_dimension(self, dimension):
        """ See abstract method. """
        self.d = dimension
        if self.d > self.d_lim:
            raise ParameterError('QRNG Korobov requires dimension <=2^31')
