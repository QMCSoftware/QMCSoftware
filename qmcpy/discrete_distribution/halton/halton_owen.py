from ..c_lib import c_lib
import ctypes
from numpy import *
from ...util import ParameterError, ParameterWarning
import warnings


class HaltonOwen(object):
    """
    References:

        [1] Owen, A. B. "A randomized Halton algorithm in R," 2017. arXiv:1706.02808 [stat.CO]
    """

    def __init__(self, dimension, generalize, randomize, seed):
        self.halton_owen_cf = c_lib.halton_owen
        self.halton_owen_cf.argtypes = [
            ctypes.c_int,  # n
            ctypes.c_int,  # d
            ctypes.c_int, # n0
            ctypes.c_int, # d0
            ctypes.c_int, # randomize
            ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # result array 
            ctypes.c_uint64]  # seed
        self.halton_owen_cf.restype = None
        self.g = generalize
        if not self.g:
            raise ParameterError("Owen Halton must be generalized")
        self.r = randomize
        self.d_lim = 1000
        self.set_dimension(dimension)
        self.set_seed(seed)

    def gen_samples(self, n_min, n_max, warn):
        if n_min == 0 and self.r==False and warn:
            warnings.warn("Non-randomized Owen Halton sequence includes the origin",ParameterWarning)
        n = int(n_max-n_min)
        x = zeros((n , self.d), dtype=double)
        self.halton_owen_cf(n, self.d, int(n_min), 0, self.r, x, self.s)
        return x
    
    def set_seed(self, seed):
        self.s = seed if seed else random.randint(2**32, dtype=uint64)
        return self.s
        
    def set_dimension(self, dimension):
        self.d = dimension
        if self.d > self.d_lim:
            raise ParameterError('Owen Halton requires dimension <= %d'%self.d_lim)
        return self.d
    
    def get_params(self):
        return self.d, self.g, self.r, self.s