import numpy as np
import os
from qmcpy.machine_learning.compression.myhosobol import MyHOSobol
from qmcpy.machine_learning.compression.computeMXY import computeWeights

def approxmeanMXY(nu, m, x, y, d):
    s = x.shape[1]
    path = os.getcwd() + os.sep
    z = MyHOSobol(m, s, d, dat_file=f'{path}sobol2.dat')
    z_transpose = np.transpose(z).copy()
    base = 2
    weights = computeWeights(nu, m, base, x, z_transpose, y)
    return weights, z


if __name__ == "__main__":

    import os
    path = os.getcwd()+os.sep+"qmcpy/machine_learning/compression"+os.sep
    x = np.loadtxt(f"test_data/reg_x.csv", delimiter=',')
    y = np.loadtxt(f"test_data/reg_y.csv", delimiter=",")
    z_true = np.loadtxt(f"test_data/reg_z.csv", delimiter=",")
    weights_true = np.loadtxt(f"test_data/reg_weights.csv",    delimiter=",")
    weights, z = approxmeanMXY(nu=1, m=6, x=x, y=y, d=1)
    np.allclose(z, z_true, atol=1e-3)
    np.allclose(weights, weights_true, atol=1e-3)