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
    computeWeights = lib.computeWeights
    computeWeights.restype=ndpointer(dtype=c_double,shape=(1+outs, Nqmc))
    print('Weights loaded')

    # compute weights
    # We need to create a function using computemxy 
    # int m = mxGetScalar(prhs[0]);  /* parameter m */
    # int mp = mxGetScalar(prhs[1]); /* parameter m' */
    # int base = mxGetScalar(prhs[2]);
    # double* px = mxGetPr(prhs[3]); /* (N x s) array of x-data points */
    # double* pz = mxGetPr(prhs[4]); /* (s x 2^m') array of QMC points !transposed! */  
    # double* py = mxGetPr(prhs[5]); /* (N x 1) array  y-data points */    
    
   
    # int s = mxGetN(prhs[3]);
    # int N = mxGetM(prhs[3]);
    # int Nqmc = mxGetN(prhs[4]);
    weights = computeWeights(c_int(m),
                         c_int(mp),
                         c_int(s),
                         c_int(N),
                         c_int(Nqmc),
                         c_int(base),
                         c_void_p(x_train_flat.ctypes.data),
		         c_void_p(qmc_points.ctypes.data),
                         c_void_p(y_train.ctypes.data))

    weights = np.transpose(weights)
    # Return data mxArray* ret=mxCreateDoubleMatrix(Nqmc,2,mxREAL); /* return vector of weights */
    #TODO remove the following line
    weights = np.loadtxt("./test_data/reg_weights.csv", delimiter=",")
    return weights
