# coding: utf-8
"""
Interfaces to quasi-random sequences in C.

References:
    
    [1] Marius Hofert and Christiane Lemieux (2019). 
    qrng: (Randomized) Quasi-Random Number Generators. 
    R package version 0.0-7.
    https://CRAN.R-project.org/package=qrng.

    [2] Faure, Henri, and Christiane Lemieux. 
    “Implementation of Irreducible Sobol’ Sequences in Prime Power Bases.” 
    Mathematics and Computers in Simulation 161 (2019): 13–22. Crossref. Web.

    [3] Owen, A. B.A randomized Halton algorithm in R2017. arXiv:1706.02808 [stat.CO]

    [4] Fischer, Gregory & Carmon, Ziv & Zauberman, Gal & L’Ecuyer, Pierre. (1999).
    Good Parameters and Implementations for Combined Multiple Recursive Random Number Generators. 
    Operations Research. 47. 159-164. 10.1287/opre.47.1.159. 
"""

import ctypes
import numpy
import os
from glob import glob

ispow2 = lambda n: (numpy.log2(n)%1) == 0.

# load library
path = os.path.dirname(os.path.abspath(__file__))
f = glob(path+'/c_lib*')[0]
lib = ctypes.CDLL(f,mode=ctypes.RTLD_GLOBAL)
# halton_owen
halton_owen_cf = lib.halton_owen
halton_owen_cf.argtypes = [
    ctypes.c_int,  # n
    ctypes.c_int,  # d
    ctypes.c_int, # n0
    ctypes.c_int, # d0
    ctypes.c_int, # randomize
    numpy.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # result array 
    ctypes.c_long]  # seed
halton_owen_cf.restype = None
# korobov_qrng
korobov_qrng_cf = lib.korobov_qrng
korobov_qrng_cf.argtypes = [
    ctypes.c_int,  # n
    ctypes.c_int,  # d
    numpy.ctypeslib.ndpointer(ctypes.c_int, flags='C_CONTIGUOUS'),  # generator
    ctypes.c_int,  # randomize
    numpy.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # result array 
    ctypes.c_long]  # seed
korobov_qrng_cf.restype = None
# halton_qrng
halton_qrng_cf = lib.halton_qrng
halton_qrng_cf.argtypes = [
    ctypes.c_int,  # n
    ctypes.c_int,  # d
    ctypes.c_int,  # generalized
    numpy.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # res
    ctypes.c_long]  # seed
halton_qrng_cf.restype = None
# sobol_qrng
sobol_qrng_cf = lib.sobol_qrng
sobol_qrng_cf.argtypes = [
    ctypes.c_int,  # n
    ctypes.c_int,  # d
    ctypes.c_int,  # randomize
    numpy.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # res
    ctypes.c_int,  # skip
    ctypes.c_int, # graycode
    ctypes.c_long]  # seed
sobol_qrng_cf.restype = None
# sobol_seq51
sobol_seq51_cf = lib.sobol_seq51
sobol_seq51_cf.argtypes = [
    ctypes.c_int, # n
    ctypes.c_int, # d
    ctypes.c_int, # skip
    numpy.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # res
]
sobol_seq51_cf.restype = None

# MRG63k3a
mrg63ka = lib.MRG63k3a
mrg63ka.argtypes = None
mrg63ka.restype = ctypes.c_double

def halton_owen(n, d, n0, d0, randomize, seed):
    # Handle input dimension correctness and corner cases
    if min(n0,d0) < 0:
        raise Exception("Starting indices (n0, d0) cannot be < 0, input had (%d,%d)"%(n0,d0))
    if min(n,d) < 0:
        raise Exception("Cannot have negative n or d")
    if n==0 or d==0: # Odd corner cases: user wants n x 0 or 0 x d matrix.
        return array([],dtype=double)
    if d0+d > 1000:
        raise Exception("Implemented only for d <= %d"%D)
    res = numpy.zeros((n, d), dtype=numpy.double)
    halton_owen_cf(n, d, n0, d0, randomize, res, seed)
    return res


def halton_qrng(n, d, generalize, seed):
    """
    Generalized Halton sequence
    
    Args:
        n (int): number of points
        d (int): dimension
        generalize (bool): string indicating which sequence is generated
            (generalized Halton (1) or (plain) Halton (0))
        seed (int): random number generator seed
    
    Returns:
        ndarray: an (n, d)-matrix containing the quasi-random sequence
    """
    if not(n >= 1 and d >= 1):
        raise Exception('ghalton_qrng input error')
    if n > (2**32 - 1):
        raise Exception('n must be <= 2^32-1')
    if d > 360:
        raise Exception('d must be <= 360')
    res = numpy.zeros((d, n), dtype=numpy.double)
    halton_qrng_cf(n, d, generalize, res, seed)
    return res.T


def korobov_qrng(n, d, generator, randomize, seed):
    """
    Korobov's sequence
    
    Args:
        n (int): number of points (>= 2 as generator has to be in {1,..,n-1}
        d (int): dimension
        generator (ndarray of ints): generator in {1,..,n-1}
            either a vector of length d
            or a single number (which is appropriately extended)
        randomize (boolean): random shift
        seed (int): random number generator seed
    
    Returns:
        ndarray: (n, d)-matrix containing the quasi-random sequence
    """
    generator = numpy.array(generator, dtype=numpy.int32)
    l = len(generator)
    if not (n >= 2 and d >= 1 and (l == 1 or l == d) and
       (generator >= 1).all() and (generator <= (n - 1)).all()):
        raise Exception('korobov_qrng input error')
    lim = 2**31 - 1
    if n > lim:
        raise Exception('n must be <=2^32-1')
    if d > lim:
        raise Exception('d must be <=2^31-1')
    if l == 1:
        generator = (generator**numpy.arange(d, dtype=numpy.int32)) % n
    res = numpy.zeros((d, n), dtype=numpy.double)
    korobov_qrng_cf(n, d, generator, randomize, res, seed)
    return res.T


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

def example_use(plot=False):
    import time
    # constants
    n = 2**11
    d = 2
    randomize = True
    seed = 7
    # generate points
    #    halton_owen
    t0 = time.time()
    halton_owen_pts = halton_owen(n, d, n0=0, d0=0, randomize=True, seed=seed)
    halton_owen_t = time.time() - t0
    #    halton_qrng
    t0 = time.time()
    halton_qrng_pts = halton_qrng(n, d, generalize=True, seed=seed)
    halton_qrng_t = time.time() - t0
    #    korobov_qrng
    t0 = time.time()
    korobov_qrng_pts = korobov_qrng(n, d, generator=[2], randomize=randomize, seed=seed)
    korobov_qrng_t = time.time() - t0
    #    sobol_qrng
    t0 = time.time()
    sobol_qrng_pts = sobol_qrng(n, d, shift=randomize, skip=0, graycode=False, seed=7)
    sobol_qrng_t = time.time() - t0
    #    sobol_seq51
    t0 = time.time()
    sobol_seq51_pts = sobol_seq51(n, d, skip=0)
    sobol_seq51_t = time.time() - t0
    #    MRG63k3a
    t0 = time.time()
    mrg63ka_pts = numpy.array([mrg63ka() for i in range(n * d)]).reshape((n, d))
    mrg63ka_t = time.time() - t0
    # outputs
    from matplotlib import pyplot
    fig, ax = pyplot.subplots(nrows=1, ncols=6, figsize=(15, 3))
    for i, (name, pts, time) in enumerate(zip(
        ['Halton_Owen',   'Halton_QRNG',   'Korobov_QRNG',   'Sobol_QRNG',    'Sobol_Seq51',   'MRG63k3a'],
        [halton_owen_pts, halton_qrng_pts, korobov_qrng_pts,  sobol_qrng_pts, sobol_seq51_pts, mrg63ka_pts],
        [halton_owen_t,   halton_qrng_t,   korobov_qrng_t,    sobol_qrng_t,   sobol_seq51_t,   mrg63ka_t])):
        print('%s Points in %.3f sec' % (name, time))
        print('\t' + str(pts).replace('\n', '\n\t'))
        if plot and d == 2:
            ax[i].scatter(pts[:, 0], pts[:, 1], s=.5)
            ax[i].set_xlim([0, 1])
            ax[i].set_ylim([0, 1])
            ax[i].set_aspect('equal')
            ax[i].set_title('%s Points' % name)
    if plot and d == 2:
        fig.suptitle('points with n=%d, d=%d, randomize=%s' % (n, d, randomize))
        fig.tight_layout()
        pyplot.show()

if __name__ == '__main__':
    example_use(plot=True)