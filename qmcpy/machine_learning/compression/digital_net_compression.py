from ctypes import *
from numpy.ctypeslib import ndpointer
from numpy import *
from qmcpy import *
import logging
from ctypes import *
import numpy as np
from approxmeanMXY import *
from numpy.ctypeslib import ndpointer
import copy


class DigitalNetDataCompressor:
    """
    Computes the weights W_X,Y and W_X.
    nu ... \nu in the paper
    m ... \ell in the paper
    s ... dimension of data
    N ... number of datapoints
    Nqmc ... number of qmc points
    outs ... output dimension of the neural network (weights W_X,Y are vector valued if outs>1)
    px ... pointer to datapoints array
    pz ... pointer to qmc points array
    py ... pointer to y values array

    Output is a pointer to a vector which contains the weights W_X (Nqmc entries), and then the dimensions of W_X,Y (Nqmc x outs entries)
    in the same order as the qmc points.


    Output is a pointer to a vector which contains the weights W_X (Nqmc entries),
    and then the dimensions of W_X,Y (Nqmc x outs entries)  in the same order as the qmc points.

    References:
    [1] J. Dick, M. Feischl, A quasi-Monte Carlo data compression algorithm for machine learning, Journal of Complexity, https://doi.org/10.1016/j.jco.2021.101587
    """

    def __init__(self, nu, m, N, Ndata, output_dimentsion, dataset):
        self.nu = 3
        self.Ndata
        self.outs = output_dimentsion
        self.dataset = dataset
        self.labels = labels
        self.sobol = sobol
        self.outputfile = outputfile

    # load c functions

    def get_dataset(self):
        return dataset

    def get_labels(self):
        return labels

    def get_sobol(self):
        return self.sobol

    def compute_weights(self, nu, m, base, z, y):
        """
        >>> X = np.loadtxt("./test_data/reg_x.csv", delimiter=',')
        >>> y = np.loadtxt("./test_data/reg_y.csv", delimiter=",")
        >>> z = np.loadtxt("./test_data/reg_z.csv", delimiter=',').T
        >>> weights = computeMXY(nu=1, m=6, base=2, x=x, z=z, y=y)
        >>> weights_true = np.loadtxt("./test_data/reg_weights.csv",  delimiter=",")
        >>> np.allclose(weights, weights_true, atol=1e-3)
        True
        """
        Nqmc = 2 ** m
        outs = 1
        s = dataset.shape[1]
        N = labels.shape[0]

        # load c functions
        lib = cdll.LoadLibrary("../c_lib/c_lib.cpython-39-darwin.so")

        # call to c function computeWeights
        computeWeights = lib.computeWeights
        computeWeights.restype = ndpointer(dtype=c_double, shape=(Nqmc * (1 + outs)))

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

        weights = weights_1d.reshape((1 + outs, Nqmc)).T

        return weight