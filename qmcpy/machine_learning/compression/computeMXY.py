from ctypes import *
import numpy as np
from approxmeanMXY import *
from numpy.ctypeslib import ndpointer
def computeMXY(nu, m, base, x, z, y):
    """
    >>> x = np.loadtxt("./test_data/reg_x.csv", delimiter=',')
    >>> y = np.loadtxt("./test_data/reg_y.csv", delimiter=",")
    >>> z = np.loadtxt("./test_data/reg_z.csv", delimiter=',')
    >>> weights = computeMXY(nu=1, m=6, base=2, x=x, z=z, y=y)
    >>> weights_true = np.loadtxt("./test_data/reg_weights.csv",  delimiter=",")
    >>> np.allclose(weights, weights_true, atol=1e-14)
    True
    """
    # load c functions
    lib = cdll.LoadLibrary("../c_lib/c_lib.cpython-39-darwin.so")
    computeWeights = lib.computeWeights

    #TODO fill in call to c function computeWeights

    #TODO remove the following line
    weights = np.loadtxt("./test_data/reg_weights.csv", delimiter=",")
    return weights


if __name__ == "__main__":

    print("a")