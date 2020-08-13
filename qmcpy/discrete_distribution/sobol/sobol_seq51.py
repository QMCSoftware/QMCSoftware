from ..c_lib import c_lib
import ctypes
from numpy import *

sobol_seq51_cf = c_lib.sobol_seq51
sobol_seq51_cf.argtypes = [
    ctypes.c_int, # n
    ctypes.c_int, # d
    ctypes.c_int, # skip
    numpy.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # res
]
sobol_seq51_cf.restype = None

def sobol_seq51(n, d, skip):
    """ 
    Sobol sequence. 

    Args:
        n (int): number of points
        d (int): dimension
        siip (int): number of inital points in the sequence to skip
    """
    max_num = 2**30-1
    max_dim = 51
    if (n+skip) > max_num or d > max_dim:
        raise Exception('SobolSeq51 supports a max of %d samples and %d dimensions'%(max_num,max_dim))
    res = numpy.zeros((n,d), dtype=numpy.double)
    sobol_seq51_cf(n, d, skip, res)
    return res
