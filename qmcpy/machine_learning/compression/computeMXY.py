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
    >>> np.allclose(weights, weights_true, atol=1e-7)
    True
    """
    Nqmc = 2**m
    outs = 1
    s = x.shape[1]
    N = y.shape[0]
    # load c functions
    lib = cdll.LoadLibrary("../c_lib/c_lib.cpython-39-darwin.so")
    computeWeights = lib.computeWeights

    #TODO fill in call to c function computeWeights
    computeWeights = lib.computeWeights
    computeWeights.restype=ndpointer(dtype=c_double,shape=(1+outs, Nqmc))

    
    # compute weights
    # We need to create a function using computemxy 
  
    
    weights = computeWeights(c_int(nu),                              
                         c_int(m),                                 
                         c_int(s),                                  
                         c_int(N),                                  
                         c_int(Nqmc),                               
                         c_int(base),                               
                         c_void_p(x.ctypes.data),         
        	         c_void_p(z.ctypes.data),           
                         c_void_p(y.ctypes.data))            
    

    weights = np.transpose(weights)
    # Return data mxArray* ret=mxCreateDoubleMatrix(Nqmc,2,mxREAL); /* return vector of weights */
    #TODO remove the following line
    #weights = np.loadtxt("./test_data/reg_weights.csv", delimiter=",")
    print(weights)
    #np.savetxt('test.out', weights, delimiter=',')
    return weights
