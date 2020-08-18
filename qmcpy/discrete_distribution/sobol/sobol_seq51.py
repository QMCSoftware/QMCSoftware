from ...util import ParameterError, ParameterWarning
from ..c_lib import c_lib
import ctypes
from numpy import *
import warnings


class SobolSeq51(object):
    """
    References:
        
        [1] I.M. Sobol', V.I. Turchaninov, Yu.L. Levitan, B.V. Shukhman: 
        "Quasi-Random Sequence Generators" Keldysh Institute of Applied Mathematics, 
        Russian Acamdey of Sciences, Moscow (1992).

        [2] Sobol, Ilya & Asotsky, Danil & Kreinin, Alexander & Kucherenko, Sergei. (2011). 
        Construction and Comparison of High-Dimensional Sobol' Generators. Wilmott. 
        2011. 10.1002/wilm.10056. 
    """

    def __init__(self, dimension, randomize, graycode, seed):
        
        self.sobol_seq51_cf = c_lib.sobol_seq51
        self.sobol_seq51_cf.argtypes = [
            ctypes.c_int, # n
            ctypes.c_int, # d
            ctypes.c_int, # skip
            ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # res
        ]
        self.sobol_seq51_cf.restype = None
        self.r = randomize
        self.g = graycode
        if self.g:
            raise ParameterError('Seq51 Sobol does not support Graycode ordering. Use "QRNG" backend for graycode ordering.')
        if self.r:
            raise ParameterError('Seq51 Sobol does not yet support randomization.')
        self.n_lim = 2**30
        self.d_lim = 51
        self.set_seed(seed)
        self.set_dimension(dimension)

    def gen_samples(self, n_min, n_max, warn):
        if n_max > self.n_lim:
            raise ParameterError("Seq51 Sobol requires n_max <= 2^30")
        if n_min == 0 and self.r==False and warn:
            warnings.warn("Non-randomized Seq51 Sobol sequence includes the origin",ParameterWarning)
        n = int(n_max-n_min)
        x = zeros((n,self.d), dtype=double)
        self.sobol_seq51_cf(n, self.d, int(n_min), x)
        return x

    def set_seed(self, seed):
        self.s = seed if seed else random.randint(2**32)
        return self.s
        
    def set_dimension(self, dimension):
        self.d = dimension
        if self.d > self.d_lim:
            raise ParameterError("Seq51 Sobol requires dimension <= %d"%self.d_lim)
        return self.d
    
    def get_params(self):
        return self.d, self.r, self.g, self.s
