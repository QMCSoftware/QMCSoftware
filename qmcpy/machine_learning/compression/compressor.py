import warnings
from numpy import *
#from ..c_lib import c_lib
import numpy as np
from ctypes import *
from numpy.ctypeslib import ndpointer
import tensorflow as tf



class compression:

        '''
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
        and then the dimensions of W_X,Y (Nqmc x outs entries)  in the same order as the qmc points.'''
        def __init__(self, nu, m, s, N, Ndata, Nqmc):
                self.m = 10 #ell in the paper
                self.nu = 3
                self.Ndata = 60000
                self.Nqmc = 2**m
                self.s = s
                self.Nqmc = Nqmc
                self.outs = output_dimentsion

                # load c functions
                lib = cdll.LoadLibrary("/home/r2q2/Projects/QMCSoftware/qmcpy/machine_learning/c_lib/computeMXY.so")
                computeWeights = lib.computeWeights
                computeWeights.restype=ndpointer(dtype=c_double,shape=(1+outs,Nqmc))

        def get_dataset(self, dataset):
                mnist = tf.keras.datasets.mnist
                return mnist

        def compute_weights(self, x_train, y_train, x_test, y_test, output_file):
                # compute weights
                # load mnist data and rescale to 10x10
                mnist = self.get_dataset('dataset')

                (x_train, y_train), (x_test, y_test) = mnist.load_data()
                x_train, x_test = x_train / 255.0, x_test / 255.0
                y_train =  np.float64(tf.one_hot(y_train,10).numpy())


                x_train=x_train[:Ndata,:,:]
                y_train = y_train[:Ndata]

                x_train = x_train[...,tf.newaxis]
                x_train = tf.image.resize(x_train,[10,10])
                x_train = x_train.numpy()[:,:,:,0]

                # data dimension
                Ndata=x_train.shape[0]
                s=x_train.shape[1]*x_train.shape[2]
                outs = y_train.shape[1]

                # flatten 10x10 to 100x1 for weight computation
                x_train_flat = np.float64(0.99*np.transpose(x_train.reshape(x_train.shape[0],s)))

                # load qmc points
                dig_net = DigitalNetB2(2,seed=6)
                qmc_points = np.array(dig_net.gen_samples(Nqmc), dtype=np.ndarray)
                #qmc_points = np.loadtxt('sobol.dat')
                #breakpoint()
                qmc_points = qmc_points[0:Nqmc,0:s]


                print(qmc_points.shape)
                print(x_train_flat.shape)

                weights = computeWeights(c_int(self.nu),
                                         c_int(self.m),
                                         c_int(self.s),
                                         c_int(self.Ndata),
                                         c_int(self.Nqmc),
                                         c_int(self.outs),
                                         c_void_p(self.x_train_flat.ctypes.data),
                                         c_void_p(self.qmc_points.ctypes.data),
                                         c_void_p(self.y_train.ctypes.data))
                weights = np.transpose(weights)
                breakpoint()