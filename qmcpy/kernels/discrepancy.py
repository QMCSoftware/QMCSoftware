import qmcpy as qp
import numpy as np
import scipy.stats
import time

def discrepancy(method, x):
    n, d = x.shape
    X_expanded = np.zeros((n,n,d)) + x
    Y = np.resize(x, (n, 1, d))
    if method == "S":
        A = (4/3)**d
        B = ((3-x**2)/2).prod(axis=1)
        C = (2 - np.maximum(X_expanded, Y)).prod(axis=2)
    elif method == "L2":
        A = (1/3)**d
        B = ((1-x**2)/2).prod(axis=1)
        C = (1 - np.maximum(X_expanded, Y)).prod(axis=2)
    elif method == "C":
        A = (13/12)**d
        B = (1 + (.5*abs(x - .5)) - (.5*((x -.5)**2))).prod(axis=1)
        C = (1 + (.5*abs(X_expanded - .5)) + (.5*abs(Y - .5)) - (.5*abs(X_expanded - Y))).prod(axis=2)
    elif method == "Sy":
        A = (4/3)**d
        B = (1 + 2*x - (2*(x**2))).prod(axis=1)
        C = (2 - (2*abs(X_expanded - Y))).prod(axis=2)
    elif method == "WA":
        A = -(4/3)**d
        B = 0
        C = (1.5 - (abs(b_1 - y)*(1 - abs(b_1 - y)))).prod(axis=2)
    elif callable(method):
        double_integral, single_integral, kernel = method(x)
        return np.sqrt(double_integral - (2*np.mean(single_integral)) + np.mean(np.mean(kernel)))
    else:
        return False
    return np.sqrt(A - (2*np.mean(B)) + np.mean(np.mean(C)))

