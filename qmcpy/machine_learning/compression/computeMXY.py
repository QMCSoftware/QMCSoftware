from ctypes import *
import numpy as np
from approxmeanMXY import *
from numpy.ctypeslib import ndpointer
import copy

def computeWeights(nu, m, base, x, z, y):
    """
    >>> x = np.loadtxt("./test_data/reg_x.csv", delimiter=',')
    >>> y = np.loadtxt("./test_data/reg_y.csv", delimiter=",")
    >>> z = np.loadtxt("./test_data/reg_z.csv", delimiter=',').T
    >>> weights = computeMXY(nu=1, m=6, base=2, x=x, z=z, y=y)
    >>> weights_true = np.loadtxt("./test_data/reg_weights.csv",  delimiter=",")
    >>> np.allclose(weights, weights_true, atol=1e-3)
    True
    """
    Nqmc = 2**m
    outs = 1
    s = x.shape[1]
    N = y.shape[0]

    #Debugging c functions
    breakpoint()
    
    # load c functions
    lib = cdll.LoadLibrary("../c_lib/c_lib.cpython-39-darwin.so")

    # call to c function computeWeights
    computeWeights = lib.computeWeights
    computeWeights.restype = ndpointer(dtype=c_double, shape=(Nqmc*(1+outs)))

    # Change the storage order of the matrices to column-major
    x2 = copy.deepcopy(np.asfortranarray(x))
    z2 = copy.deepcopy(np.asfortranarray(z))

    # compute weights
    weights_1d = computeWeights(c_int(nu),
                             c_int(m),
                             c_int(s),
                             c_int(N),
                             c_int(Nqmc),
                             c_void_p(x2.ctypes.data),
                             c_void_p(z2.ctypes.data),
                             c_void_p(y.ctypes.data))

    weights = weights_1d.reshape((1+outs, Nqmc)).T

    return weights
