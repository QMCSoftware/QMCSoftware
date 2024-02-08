from numpy import array, ndarray, log2
import numpy as np
from copy import copy

def K(x,y):
    d = x.np.shape
    prod = 1
    for k in range(0, d):
        prod = prod * (2 - max(x[k],y[k]))
    return prod

def discrepancy(x):
    (n,d) = x.np.shape
    A = (4/3)**d
    B = 0 
    for i in range(0, n):
        x_i = x[i]
        prod = 1
        for k in range(0,d):
            prod = prod * (3 - (x_i[k]**2))/2
        B = B + prod
    C = 0
    for i in range(0, n):
        for j in range(0, n):
            x_i = x[i]
            x_j = x[j]
            C = C + K(x_i, x_j, n)

    value = A - ((2/n)*B) + ((n**(-2)) * C)
    return value**.5
