from ...util import ParameterError, ParameterWarning
from ..c_lib import c_lib
import ctypes
from numpy import *
import warnings


class SobolQRNG(object):
    """
    References:

        [1] Marius Hofert and Christiane Lemieux (2019). 
        qrng: (Randomized) Quasi-Random Number Generators. 
        R package version 0.0-7.
        https://CRAN.R-project.org/package=qrng.

        [2] Faure, Henri, and Christiane Lemieux. 
        “Implementation of Irreducible Sobol' Sequences in Prime Power Bases.” 
        Mathematics and Computers in Simulation 161 (2019): 13–22. Crossref. Web.
    """

    def __init__(self, dimension, randomize, graycode, seed):
        self.sobol_qrng_cf = c_lib.sobol_qrng
        self.sobol_qrng_cf.argtypes = [
            ctypes.c_int,  # n
            ctypes.c_int,  # d
            ctypes.c_int,  # randomize
            ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # res
            ctypes.c_int,  # skip
            ctypes.c_int, # graycode
            ctypes.c_long]  # seed
        self.sobol_qrng_cf.restype = None
        self.r = randomize
        self.g = graycode
        self.n_lim = 2**31
        self.d_lim = 16510
        self.set_seed(seed)
        self.set_dimension(dimension)

    def gen_samples(self, n_min, n_max, warn):
        if n_max > self.n_lim:
            raise ParameterError("QRNG Sobol requires n_max <= 2^31")
        if n_min == 0 and self.r==False and warn:
            warnings.warn("Non-randomized QRNG Sobol sequence includes the origin",ParameterWarning)
        if (not self.g) and not ( (n_min==0 or log2(n_min)%1==0) and log2(n_max)%1==0 ):
            raise ParameterError('''
                QRNG Sobol with standard (non-graycode) ordering requires
                    n_min is 0 or a power of 2 and 
                    (n_max) is a power of 2''')
        n = int(n_max-n_min)
        x = zeros((self.d,n), dtype=double)
        self.sobol_qrng_cf(n, self.d, self.r, x, int(n_min), self.g, self.s)
        return x.T

    def set_seed(self, seed):
        self.s = seed if seed else random.randint(2**32)
        return self.s
        
    def set_dimension(self, dimension):
        self.d = dimension
        if self.d > self.d_lim:
            raise ParameterError("QRNG Sobol requires dimension <= %d"%self.d_lim)
        return self.d
    
    def get_params(self):
        return self.d, self.r, self.g, self.s