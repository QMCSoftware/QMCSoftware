import numpy as np
import pickle
from ctypes import *
from numpy.ctypeslib import ndpointer
import tensorflow as tf
#from qmcpy import *

m = 10
nu = 3

Ndata = 60000
Nqmc = 2**m;

# load mnist data and rescale to 10x10
mnist = tf.keras.datasets.mnist
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

#dig_net = DigitalNetB2(2,seed=6)
#qmc_points = np.array(dig_net.gen_samples(Nqmc), dtype=np.ndarray)
qmc_points = np.loadtxt('sobol.dat')
#breakpoint()
qmc_points = qmc_points[0:Nqmc,0:s]


print(qmc_points.shape)
print(x_train_flat.shape)

# load c functions
lib = cdll.LoadLibrary("/home/r2q2/Projects/QMCSoftware/qmcpy/machine_learning/c_lib/computeMXY.so")
computeWeights = lib.computeWeights
computeWeights.restype=ndpointer(dtype=c_double,shape=(1+outs,Nqmc))
print('Weights loaded')

# compute weights
weights = computeWeights(c_int(nu),c_int(m),c_int(s),c_int(Ndata),c_int(Nqmc),c_int(outs),c_void_p(x_train_flat.ctypes.data),
		c_void_p(qmc_points.ctypes.data),c_void_p(y_train.ctypes.data))
weights = np.transpose(weights)

breakpoint()

with open('weights.pkl', 'wb') as handle:
    pickle.dump(weights, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
