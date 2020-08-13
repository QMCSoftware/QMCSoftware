from ..c_lib import c_lib
from ...util import ParameterError, ParameterWarning
import ctypes
from numpy import *
import warnings


class KorobovQRNG(object):
    """"
    References:

        [1] Marius Hofert and Christiane Lemieux (2019). 
        qrng: (Randomized) Quasi-Random Number Generators. 
        R package version 0.0-7.
        https://CRAN.R-project.org/package=qrng.
    """
    
    def __init__(self, dimension, generator, randomize, seed):
        self.korobov_qrng_cf = c_lib.korobov_qrng
        self.korobov_qrng_cf.argtypes = [
            ctypes.c_int,  # n
            ctypes.c_int,  # d
            numpy.ctypeslib.ndpointer(ctypes.c_int, flags='C_CONTIGUOUS'),  # generator
            ctypes.c_int,  # randomize
            numpy.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # result array 
            ctypes.c_long]  # seed
        self.korobov_qrng_cf.restype = None
        self.g = array(generator, dtype=int)
        self.l = len(self.g)
        if (self.l!=1 and self.l!=self.d) or (generator<1).any() or (generator>=n).any():
            raise ParameterError('QRNG Korobov requires all(1 <= generator ints <= (n-1)) and (len(generator)=dimension or 1).')
        self.r = randomize
        self.n_lim = 2*31-1
        self.d_lim = self.n_lim
        self.set_dimension(dimension)
        self.set_seed(seed)

    def gen_samples(self, n_min, n_max, warn=True):
        if self.r==False and warn:
            warnings.warn("Non-randomized Korobov sequence includes the origin.",ParameterWarning)
        if n_min>0:
            raise ParameterError('QRNG Korobov does not support skipping samples with n_min>0.')
        if n_max < 2:
            raise ParameterError('QRNG Korobov requires n>=2.')
        if n_max > self.n_lim:
            raise Exception('QRNG Korobov requires n must be <=2^32-1')
        if self.l == 1:
            self.g = (self.g**arange(self.d, dtype=int)) % n_max
        x = zeros((d, n_max), dtype=double)
        self.korobov_qrng_cf(int(n_max), int(self.d), generator, self.r, x, self.s)
        return x.T

    def set_seed(self, seed):
        self.s = seed if seed else random.randint(2**32)
        
    def set_dimension(self, dimension):
        self.d = dimension
        if self.d > self.d_lim:
            raise ParameterError('QRNG Korobov requires dimension <=2^31-1')