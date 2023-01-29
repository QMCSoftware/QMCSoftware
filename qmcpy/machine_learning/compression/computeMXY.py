from ctypes import *
import numpy as np
from approxmeanMXY import *
from numpy.ctypeslib import ndpointer
def computeMXY(nu, m, base, x, z, y):
    # load c functions
    lib = cdll.LoadLibrary("../c_lib/c_lib.cpython-39-darwin.so")
    computeWeights = lib.computeWeights

    #TODO fill in call to c function computeWeights

    #TODO remove the following line
    weights = np.loadtxt("./test_data/reg_weights.csv", delimiter=",")
    return weights


if __name__ == "__main__":

    print("a")