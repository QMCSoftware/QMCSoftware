import ctypes
import numpy
import os
from glob import glob

ispow2 = lambda n: (numpy.log2(n)%1) == 0.

# load library
path = os.path.dirname(os.path.abspath(__file__))
f = glob(path+'/gail_lattice*')[0]
lib = ctypes.CDLL(f,mode=ctypes.RTLD_GLOBAL)
# gail lattice
lattice_f = lib.lattice
lattice_f.argtypes = [
    ctypes.c_int,  # n
    ctypes.c_int,  # d
    ctypes.c_int,  # skip
    numpy.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS')]  # x
lattice_f.restype = None
def gail_lattice_gen_2(n_min,n_max,d):
    """
    Generate d dimensionsal lattice samples from n_min to n_max
    
    Args:
        d (int): dimension of the problem, 1<=d<=100.
        n_min (int): minimum index. Must be 0 or n_max/2
        n_max (int): maximum index (not inclusive)
    """
    lgv = 600
    if d > lgv:
        raise Exception('GAIL Lattice has max dimensions %d'%lgv)
    if n_max > 2**20:
        raise Exception('GAIL Lattice has maximum points 2^20')    
    n = int(n_max-n_min)
    x = numpy.zeros((n, d), dtype=numpy.double)
    lattice_f(n, d, int(n_min), x)
    return x

if __name__ == '__main__':
    import time
    t0 = time.perf_counter()
    x = gail_lattice_gen_2(0,2**20,2)
    print('Time = %.4f'%(time.perf_counter()-t0))