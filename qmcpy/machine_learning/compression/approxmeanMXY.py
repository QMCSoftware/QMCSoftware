from myhosobol import MyHOSobol
import numpy as np
from computeMXY import *
def approxmeanMXY(nu, m, x, y, d):
    """"

    >>> x = np.loadtxt("./test_data/reg_x.csv", delimiter=',')
    >>> y = np.loadtxt("./test_data/reg_y.csv", delimiter=",")
    >>> weights, z = approxmeanMXY(nu=1, m=6, d=1, x=x,y=y)
    >>> z_true  = np.loadtxt("./test_data/reg_z.csv", delimiter=",")
    >>> weights_true = np.loadtxt("./test_data/reg_weights.csv",    delimiter=",")
    >>> np.allclose(weights, weights_true, atol=1e-14)
    True
    """

    s = x.shape[1]
    z = MyHOSobol(m, s, d)
    base = 2
    weights = computeMXY(nu, m, base, x, z, y)
    return weights, z