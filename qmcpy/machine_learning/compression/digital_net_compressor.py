from qmcpy.machine_learning.c_lib import c_lib
from ctypes import *
import numpy as np
from numpy.ctypeslib import ndpointer
import copy
from qmcpy import Sobol


class DigitalNetDataCompressor:
    """
    Computes the weights $W_X$ and $W_{X,Y}$.

    Args:
        nu (int): $\nu$ in the paper
        m (int): $\ell$ in the paper
        dataset (ndarray): $X$ in the paper
        labels (ndarray): $Y$ in the paper
        sobol (ndarray): $2**m$ Sobol sampling points in d-dimensional unit cube
        alpha (int):

    Return:
        weights: first column is $W_X$ and second column is $W_{X,Y}$

    Reference:
    [1] J. Dick and M. Feischl, A quasi-Monte Carlo data compression algorithm for
        machine learning, Journal of Complexity, https://doi.org/10.1016/j.jco.2021.101587
    """

    def __init__(self, nu, m, dataset, labels, sobol=None, alpha=1):
        # inputs
        self.nu = nu
        self.m = m
        self.dataset = dataset
        self.labels = labels
        self.alpha = alpha
        self.sobol = sobol

        # derived
        self.s = self.dataset.shape[1]

        # outputs
        self.weights = None

    def approx_mean_mxy(self):
        if self.sobol is None:
            lds = Sobol(self.s)
            self.sobol = lds.gen_samples(2 ** self.m)
        self.compute_weights()

    def compute_weights(self):
        n_qmc = 2 ** self.m
        outs = 1
        s = self.dataset.shape[1]
        n = self.labels.shape[0]
        self.sobol = self.sobol[:2 ** self.m, :self.s]

        # call to c function computeWeights
        compute_weights = c_lib.computeWeights
        compute_weights.restype = ndpointer(dtype=c_double, shape=(n_qmc * (1 + outs)))

        # Change the storage order of the matrices to column-major
        x2 = copy.deepcopy(np.asfortranarray(self.dataset))
        sobol_t = np.transpose(self.sobol).copy()
        z2 = copy.deepcopy(np.asfortranarray(sobol_t))

        # compute weights
        '''
        nu: $\nu$ in the paper
        m: $\ell$ in the paper
        s: dimension of data
        n: number of datapoints
        n_qmc: number of qmc points
        outs: output dimension of the neural network (weights W_X,Y are vector valued if outs>1)
        px: pointer to datapoints array
        pz: pointer to qmc points array
        py: pointer to labels values array
    
        Output is a pointer to a vector which contains the weights W_X (n_qmc entries),
        and then the dimensions of W_X,Y (n_qmc x outs entries)  in the same order as the qmc points.
        '''
        weights_1d = compute_weights(c_int(self.nu),
                                     c_int(self.m),
                                     c_int(s),
                                     c_int(n),
                                     c_int(n_qmc),
                                     c_void_p(x2.ctypes.data),
                                     c_void_p(z2.ctypes.data),
                                     c_void_p(self.labels.ctypes.data))

        self.weights = copy.deepcopy(weights_1d.reshape((1 + outs, n_qmc)).T)