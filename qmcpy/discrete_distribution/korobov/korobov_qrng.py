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
            ctypeslib.ndpointer(ctypes.c_int, flags='C_CONTIGUOUS'),  # generator
            ctypes.c_int,  # randomize
            ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # result array 
            ctypes.c_long]  # seed
        self.korobov_qrng_cf.restype = None
        self.g_og = array(generator, dtype=int32)
        self.r = randomize
        self.n_lim = 2**31
        self.d_lim = self.n_lim
        self.set_dimension(dimension)
        self.set_seed(seed)

    def gen_samples(self, n_min, n_max, warn=True):
        l = len(self.g_og)
        if l == 1:
            self.g = ((self.g_og**arange(self.d, dtype=int32)) % n_max).astype(int32)
        elif l == self.d:
            self.g = self.g_og
        else:
            raise ParameterError("QRNG Korobov must have generator of length 1 or dimension.")
        if (self.g<1).any() or (self.g>=n_max).any():
            raise ParameterError('QRNG Korobov requires all(1 <= generator ints <= (n-1)).')
        if self.r==False and warn:
            warnings.warn("Non-randomized Korobov sequence includes the origin.",ParameterWarning)
        if n_min>0:
            raise ParameterError('QRNG Korobov does not support skipping samples with n_min>0.')
        if n_max < 2:
            raise ParameterError('QRNG Korobov requires n>=2.')
        if n_max > self.n_lim:
            raise Exception('QRNG Korobov requires n must be <=2^32')
        x = zeros((self.d, n_max), dtype=double)
        self.korobov_qrng_cf(int(n_max), int(self.d), self.g, self.r, x, self.s)
        return x.T

    def set_seed(self, seed):
        self.s = seed if seed else random.randint(2**32, dtype=uint64)
        return self.s
        
    def set_dimension(self, dimension):
        self.d = dimension
        if self.d > self.d_lim:
            raise ParameterError('QRNG Korobov requires dimension <=2^31')
        return self.d
    
    def get_params(self):
        return self.d, self.g_og, self.r, self.s