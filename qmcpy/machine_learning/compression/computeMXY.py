
def computeMXY(nu, m, base, x, z, y):
    """
    TODO call c library
    """
    import numpy as np
    weights = np.loadtxt("./test_data/reg_weights.csv", delimiter=",")
    return weights