from ..c_lib import c_lib
import ctypes
from numpy import *

sobol_qrng_cf = c_lib.sobol_qrng
sobol_qrng_cf.argtypes = [
    ctypes.c_int,  # n
    ctypes.c_int,  # d
    ctypes.c_int,  # randomize
    ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # res
    ctypes.c_int,  # skip
    ctypes.c_int, # graycode
    ctypes.c_long]  # seed
sobol_qrng_cf.restype = None

def sobol_qrng(n, d, shift, skip, graycode, seed):
    """
    Sobol sequence

    Args:
        n (int): number of points
        d (int): dimension
        shift (boolean): apply digital shift
        skip (int): number of initial points in the sequence to skip.
        graycode (bool): indicator to use graycode ordering (True) or natural ordering (False)
        seed (int): random number generator seed

    Returns:
        ndarray: an (n, d)-matrix containing the quasi-random sequence
    """
    if not (n >= 1 and d >= 1 and skip >= 0):
        raise Exception('sobol_qrng input error')
    if n > (2**31 - 1):
        raise Exception('n must be <= 2^32-1')
    if (not graycode) and not ( (skip==0 or ispow2(skip)) and ispow2(skip+n) ):
        raise Exception('''
            Using natural (non-graycode) ordering requires
                skip is 0 or a power of 2
                (skip+n) is a power of 2''')
    if d > 16510:
        raise Exception('d must be <= 16510')
    res = numpy.zeros((d, n), dtype=numpy.double)
    sobol_qrng_cf(n, d, shift, res, skip, graycode, seed)
    return res.T