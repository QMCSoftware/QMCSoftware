from .._discrete_distribution import DiscreteDistribution
from ...util import ParameterError
from numpy import *
from ..c_lib import c_lib
import ctypes


class Halton(DiscreteDistribution):
    """
    Quasi-Random Halton nets.

    >>> h = Halton(2,seed=7)
    >>> h.gen_samples(4)
    array([[0.166, 0.363],
           [0.666, 0.696],
           [0.416, 0.03 ],
           [0.916, 0.474]])
    >>> h.gen_samples(1)
    array([[0.166, 0.363]])
    >>> h
    Halton (DiscreteDistribution Object)
        d               2^(1)
        generalize      1
        randomize       1
        seed            7
        mimics          StdUniform
    
    References:

        [1] Marius Hofert and Christiane Lemieux (2019). 
        qrng: (Randomized) Quasi-Random Number Generators. 
        R package version 0.0-7.
        https://CRAN.R-project.org/package=qrng.
        
        [2] Owen, A. B. "A randomized Halton algorithm in R," 2017. arXiv:1706.02808 [stat.CO]
    """

    def __init__(self, dimension=1, generalize=True, randomize=True, seed=None):
        """
        Args:
            dimension (int): dimension of samples
            generalize (bool): generalize the Halton sequence?
            randomize (bool/str): If False, does not randomize Halton points. 
                If True, will use 'QRNG' randomization as in [1]. Supports max dimension 360. 
                You can also set radnomize='QRNG' or randomize='Halton' to explicitly select a randomization method. 
                Halton OWEN supports up to 1000 dimensions.
            seed (int): seed the random number generator for reproducibility
        
        Note:
            See References [1] and [2] for specific randomization methods and differences. 
        """
        self.parameters = ['d','generalize','randomize','seed','mimics']
        if isinstance(randomize,bool):
            self.backend = 'QRNG' if randomize else 'OWEN'
            self.randomize = randomize
        elif isinstance(randomize,str):
            self.backend = randomize.upper()
            self.randomize = True
        else:
            s = "Halton randomize must be True/False or 'QRNG'/'Owen'"
            raise ParameterError(s)
        self.generalize = generalize
        if self.generalize==False and self.backend=='OWEN':
            raise ParameterError("Owen halton Must be genralized")
        if self.backend=='QRNG':
            self.halton_cf = c_lib.halton_qrng
            self.halton_cf.argtypes = [
                ctypes.c_int,  # n
                ctypes.c_int,  # d
                ctypes.c_int, # n0
                ctypes.c_int,  # generalized
                ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # res
                ctypes.c_long]  # seed
            self.halton_cf.restype = None
            self.g = generalize
            self.r = randomize
            self.d_lim = 360
        elif self.backend=='OWEN':
            self.halton_cf = c_lib.halton_owen
            self.halton_cf.argtypes = [
                ctypes.c_int,  # n
                ctypes.c_int,  # d
                ctypes.c_int, # n0
                ctypes.c_int, # d0
                ctypes.c_int, # randomize
                ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # result array 
                ctypes.c_long]  # seed
            self.halton_cf.restype = None
            self.r = randomize
            self.d_lim = 1000
        else:
            s = "Halton randomize must be True/False or 'QRNG'/'Owen'"
            raise ParameterError(s)
        self.n_lim = 2**32
        self._set_dimension(dimension)
        self.set_seed(seed)
        self.low_discrepancy = True
        self.mimics = 'StdUniform'
        super(Halton,self).__init__()

    def gen_samples(self, n=None, n_min=0, n_max=8, warn=True):
        """
        Generate samples

        Args:
            n (int): if n is supplied, generate from n_min=0 to n_max=n samples. 
                Otherwise use the n_min and n_max explicitly supplied as the following 2 arguments
            n_min (int): Starting index of sequence.
            n_max (int): Final index of sequence.

        Returns:
            ndarray: (n_max-n_min) x d (dimension) array of samples
        """
        if n:
            n_min = 0
            n_max = n
        if n_max > self.n_lim:
            raise ParameterWarning("Halton requires n_max <= 2^32.")
        n = int(n_max-n_min)
        if self.backend=='QRNG':
            x = zeros((self.d, n), dtype=double)
            self.halton_cf(n, self.d, int(n_min), self.generalize, x, self.seed)
            return x.T
        elif self.backend=='OWEN':
            x = zeros((n,self.d), dtype=double)
            self.halton_cf(n, self.d, int(n_min), 0, self.randomize, x, self.seed)
            return x

    def pdf(self, x):
        return ones(x.shape[0], dtype=float)
        
    def set_seed(self, seed):
        self.seed = seed if seed else random.randint(1, 100000, dtype=uint64)
        
    def _set_dimension(self, dimension):
        self.d = dimension
        if self.d > self.d_lim:
            s = '''
                Halton with randomize='QRNG' backend supports dimension <= 360.
                Halton with randomize='OWEN' backend supports dimension <= 1000. 
                '''
            raise ParameterError(s)        
