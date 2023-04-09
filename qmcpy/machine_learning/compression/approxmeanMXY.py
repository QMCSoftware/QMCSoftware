from myhosobol import MyHOSobol
import numpy as np
from computeMXY import computeWeights
def approxmeanMXY(nu, m, x, y, d):
    """"
    >>> x = np.loadtxt("./test_data/reg_x.csv", delimiter=',')
    >>> y = np.loadtxt("./test_data/reg_y.csv", delimiter=",")
    >>> z_true  = np.loadtxt("./test_data/reg_z.csv", delimiter=",")
    >>> weights_true = np.loadtxt("./test_data/reg_weights.csv",    delimiter=",")
    >>> weights, z = approxmeanMXY(nu=1, m=6, x=x, y=y, d=1)
    >>> np.allclose(z, z_true, atol=1e-3)
    True
    >>> np.allclose(weights, weights_true, atol=1e-3)
    True
    """

    s = x.shape[1]
    z = MyHOSobol(m, s, d)
    z_transpose = np.transpose(z).copy()
    base = 2
    weights = computeWeights(nu, m, base, x, z_transpose, y)
    return weights, z
