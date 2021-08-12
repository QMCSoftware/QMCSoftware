from ._discrete_distribution import DiscreteDistribution
from ..util import ParameterError,ParameterWarning
from numpy import *
from .c_lib import c_lib
import ctypes
import warnings


class Halton(DiscreteDistribution):
    """
    Quasi-Random Halton nets.

    >>> h = Halton(2,seed=7)
    >>> h.gen_samples(4)
    array([[0.35362988, 0.38733489],
           [0.85362988, 0.72066823],
           [0.10362988, 0.05400156],
           [0.60362988, 0.498446  ]])
    >>> h.gen_samples(1)
    array([[0.35362988, 0.38733489]])
    >>> h
    Halton (DiscreteDistribution Object)
        d               2^(1)
        dvec            [0 1]
        randomize       1
        generalize      1
        entropy         7
        spawn_key       ()
    
    References:

        [1] Marius Hofert and Christiane Lemieux (2019). 
        qrng: (Randomized) Quasi-Random Number Generators. 
        R package version 0.0-7.
        https://CRAN.R-project.org/package=qrng.
    """
    # QRNG
    halton_cf_qrng = c_lib.halton_qrng
    halton_cf_qrng.argtypes = [
        ctypes.c_int,  # n
        ctypes.c_int,  # d
        ctypes.c_int, # n0
        ctypes.c_int, # generalized
        ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'), # res
        ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'), # randu_d_32
        ctypeslib.ndpointer(ctypes.c_int, flags='C_CONTIGUOUS')]  # dvec
    halton_cf_qrng.restype = None

    def __init__(self, dimension=1, randomize=True, generalize=True, seed=None):
        """
        Args:
            dimension (int or ndarray): dimension of the generator. 
                If an int is passed in, use sequence dimensions [0,...,dimensions-1].
                If a ndarray is passed in, use these dimension indices in the sequence. 
            generalize (bool): generalize flag
            randomize (bool/str): If True, apply randomization from QRNG [1], otherwise leave Halton unrandomized
            seed (int): seed the random number generator for reproducibility
        """
        self.parameters = ['dvec','randomize','generalize']
        self.randomize = randomize
        self.generalize = generalize
        self.d_max = 360
        self.mimics = 'StdUniform'
        self.low_discrepancy = True
        self.n_lim = 2**32
        super(Halton,self).__init__(dimension,seed)
        self.randu_d_32 = self.rng.uniform(size=(self.d,32)) if self.randomize else zeros((self.d,32),dtype=double)
    
    def gen_samples(self, n=None, n_min=0, n_max=8, warn=True):
        """
        Generate samples

        Args:
            n (int): if n is supplied, generate from n_min=0 to n_max=n samples. 
                Otherwise use the n_min and n_max explicitly supplied as the following 2 arguments
            n_min (int): Starting index of sequence.
            n_max (int): Final index of sequence.
            return_unrandomized (bool): return samples without randomization as 2nd return value. 
                Will not be returned if randomize=False. 

        Returns:
            ndarray: (n_max-n_min) x d (dimension) array of samples
        """
        if n:
            n_min = 0
            n_max = n
        if n_max > self.n_lim:
            raise ParameterWarning("Halton requires n_max <= 2^32.")
        if n_min == 0 and self.randomize==False and warn:
            warnings.warn("Non-randomized Halton sequence includes the origin",ParameterWarning)
        n = int(n_max-n_min)
        x = zeros((self.d,n),dtype=double)
        self.halton_cf_qrng(n,self.d,int(n_min),self.generalize,x,self.randu_d_32,int32(self.dvec))
        return x.T

    def pdf(self, x):
        return ones(x.shape[0], dtype=float)

    def _spawn(self, s, child_seeds, dimensions):
        return [
            Halton(
                dimension=dimensions[i],
                generalize=self.generalize,
                randomize=self.randomize,
                seed=child_seeds[i])
            for i in range(s)]
