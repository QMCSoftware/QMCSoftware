""" Interfaces to quasi-random sequences in C.

References
    Marius Hofert and Christiane Lemieux (2019). 
    qrng: (Randomized) Quasi-Random Number Generators. 
    R package version 0.0-7.
    https://CRAN.R-project.org/package=qrng.

    Faure, Henri, and Christiane Lemieux. 
    “Implementation of Irreducible Sobol’ Sequences in Prime Power Bases.” 
    Mathematics and Computers in Simulation 161 (2019): 13–22. Crossref. Web.
"""
import ctypes
import numpy
import os
from glob import glob

ispow2 = lambda n: (numpy.log2(n)%1) == 0.

# load library
path = os.path.dirname(os.path.abspath(__file__))
f = glob(path+'/qrng_lib*')[0]
lib = ctypes.CDLL(f,mode=ctypes.RTLD_GLOBAL)
# MRG63k3a
mrg63ka_f = lib.MRG63k3a
mrg63ka_f.argtypes = None
mrg63ka_f.restype = ctypes.c_double
# korobov
korobov_f = lib.korobov
korobov_f.argtypes = [
    ctypes.c_int,  # n
    ctypes.c_int,  # d
    numpy.ctypeslib.ndpointer(ctypes.c_int, flags='C_CONTIGUOUS'),  # generator
    ctypes.c_int,  # randomize
    numpy.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # res
    ctypes.c_long]  # seed
korobov_f.restype = None
# ghalton
ghalton_f = lib.ghalton
ghalton_f.argtypes = [
    ctypes.c_int,  # n
    ctypes.c_int,  # d
    ctypes.c_int,  # generalized
    numpy.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # res
    ctypes.c_long]  # seed
ghalton_f.restype = None
# sobol
sobol_f = lib.sobol
sobol_f.argtypes = [
    ctypes.c_int,  # n
    ctypes.c_int,  # d
    ctypes.c_int,  # randomize
    numpy.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # res
    ctypes.c_int,  # skip
    ctypes.c_int, # graycode
    ctypes.c_long]  # seed
sobol_f.restype = None


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
    
    Return:
        result (ndarray): (n, d)-matrix containing the quasi-random sequence
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
    korobov_f(n, d, generator, randomize, res, seed)
    return res.T


def ghalton_qrng(n, d, generalize, seed):
    """
    Generalized Halton sequence
    
    Args:
        n (int): number of points
        d (int): dimension
        generalize (bool): string indicating which sequence is generated
            (generalized Halton (1) or (plain) Halton (0))
        seed (int): random number generator seed
    
    Return:
        res (ndarray): an (n, d)-matrix containing the quasi-random sequence
    """
    if not(n >= 1 and d >= 1):
        raise Exception('ghalton_qrng input error')
    if n > (2**32 - 1):
        raise Exception('n must be <= 2^32-1')
    if d > 360:
        raise Exception('d must be <= 360')
    res = numpy.zeros((d, n), dtype=numpy.double)
    ghalton_f(n, d, generalize, res, seed)
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

    Return:
        res (ndarray): an (n, d)-matrix containing the quasi-random sequence
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
    sobol_f(n, d, shift, res, skip, graycode, seed)
    return res.T

def qrng_example_use(plot=False):
    import time
    # constants
    n = 2**11
    d = 2
    randomize = True
    seed = 7
    # generate points
    #    MRG63k3a
    t0 = time.perf_counter()
    mrg63ka_pts = numpy.array([mrg63ka_f() for i in range(n * d)]).reshape((n, d))
    mrg63ka_t = time.perf_counter() - t0
    #    korobov
    t0 = time.perf_counter()
    korobov_pts = korobov_qrng(n, d, generator=[2], randomize=randomize, seed=seed)
    korobov_t = time.perf_counter() - t0
    #    ghalton
    t0 = time.perf_counter()
    ghalton_pts = ghalton_qrng(n, d, generalize=True, seed=seed)
    ghalton_t = time.perf_counter() - t0
    #    sobol
    t0 = time.perf_counter()
    sobol_pts = sobol_qrng(n, d, shift=randomize, skip=0, graycode=False, seed=7)
    sobol_t = time.perf_counter() - t0
    # outputs
    from matplotlib import pyplot
    fig, ax = pyplot.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for i, (name, pts, time) in enumerate(zip(
        ['MRG63k3a', 'Korobov', 'GHalton', 'Sobol'],
        [mrg63ka_pts, korobov_pts, ghalton_pts, sobol_pts],
            [mrg63ka_t, korobov_t, ghalton_t, sobol_t])):
        print('%s Points in %.3f sec' % (name, time))
        print('\t' + str(pts).replace('\n', '\n\t'))
        if plot and d == 2:
            ax[i].scatter(pts[:, 0], pts[:, 1], s=.5)
            ax[i].set_xlim([0, 1])
            ax[i].set_ylim([0, 1])
            ax[i].set_aspect('equal')
            ax[i].set_title('%s Points' % name)
    if plot and d == 2:
        fig.suptitle('qrng points with n=%d, d=%d, randomize=%s' % (n, d, randomize))
        fig.tight_layout()
        pyplot.show()

if __name__ == '__main__':
    qrng_example_use(plot=True)