from qmcpy.machine_learning.c_lib import c_lib
from ctypes import *
from qmcpy.machine_learning.compression.myhosobol import MyHOSobol
from numpy import *
from ctypes import *
import numpy as np
from numpy.ctypeslib import ndpointer
import copy
import os


class DigitalNetDataCompressor:
    """
    Computes the weights W_X,Y and W_X.
    nu: $\nu$ in the paper
    m: $\ell$ in the paper
    s: dimension of data
    N: number of datapoints
    nqmc: number of qmc points
    outs: output dimension of the neural network (weights W_X,Y are vector valued if outs>1)
    px: pointer to datapoints array
    pz: pointer to qmc points array
    py: pointer to labels values array

    Output is a pointer to a vector which contains the weights W_X (nqmc entries), and then the dimensions of W_X,Y (Nqmc x outs entries)
    in the same order as the qmc points.


    Output is a pointer to a vector which contains the weights W_X (nqmc entries),
    and then the dimensions of W_X,Y (nqmc x outs entries)  in the same order as the qmc points.

    References:
    [1] J. Dick, M. Feischl, A quasi-Monte Carlo data compression algorithm for machine learning, Journal of Complexity, https://doi.org/10.1016/j.jco.2021.101587
    """

    def __init__(self, nu, m, dataset, labels, alpha=1):
        self.nu = nu
        self.m = m
        self.dataset = dataset
        self.labels = labels
        self.alpha = alpha
        self.sobol = None
        self.weights = None
        # self.outputfile = outputfile

    def get_dataset(self):
        return self.dataset

    def get_labels(self):
        return self.labels

    def get_sobol(self):
        return self.sobol

    def approx_mean_mxy(self):
        s = self.dataset.shape[1]
        path = os.getcwd() + os.sep
        self.sobol = MyHOSobol(self.m, s, self.alpha, dat_file=f'{path}sobol2.dat')
        z_transpose = np.transpose(self.sobol).copy()
        self.sobol_T = z_transpose
        self.compute_weights()

    def compute_weights(self):
        nqmc = 2 ** self.m
        outs = 1
        s = self.dataset.shape[1]
        N = self.labels.shape[0]

        # call to c function computeWeights
        compute_weights = c_lib.computeWeights
        compute_weights.restype = ndpointer(dtype=c_double, shape=(nqmc * (1 + outs)))

        # Change the storage order of the matrices to column-major
        x2 = copy.deepcopy(np.asfortranarray(self.dataset))
        z2 = copy.deepcopy(np.asfortranarray(self.sobol_T))

        # compute weights
        weights_1d = compute_weights(c_int(self.nu),
                                     c_int(self.m),
                                     c_int(s),
                                     c_int(N),
                                     c_int(nqmc),
                                     c_void_p(x2.ctypes.data),
                                     c_void_p(z2.ctypes.data),
                                     c_void_p(self.labels.ctypes.data))

        self.weights = weights_1d.reshape((1 + outs, nqmc)).T