from ...util import ParameterError, ParameterWarning
from ..c_lib import c_lib
import ctypes
from numpy import *
import warnings


class SobolAGS(object):

    def __init__(self, dimension, randomize, graycode, seed):
        self.sobol_ags_cf = c_lib.sobol_ags
        self.sobol_ags_cf.argtypes = [
            ctypes.c_ulong,  # n
            ctypes.c_uint32,  # d
            ctypes.c_ulong, # n0
            ctypes.c_uint32, # d0
            ctypes.c_uint8,  # randomize
            ctypes.c_uint8, # graycode
            ctypeslib.ndpointer(ctypes.c_uint32), # seeds
            ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # x (result)
            ctypes.c_uint32, # d_max
            ctypes.c_uint32, # m_max
            ctypeslib.ndpointer(ctypes.c_ulong, flags='C_CONTIGUOUS'),  # z (generating matrix)
            ctypes.c_uint8] # msb


        self.sobol_ags_cf.restype = ctyps.c_uint8
        self.r = randomize
        self.g = graycode
        self.n_lim = 2**32
        self.d_lim = 21201
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