import numpy as np
import time
from copy import copy

def discrepancy(method, x, weight = 1):
    n, d = x.shape                              #Finds the number of samples and the dimensions for our x_i's
    #find a way to generate the weight, and generate the scalar into an array
    #For L2 star, the weighted kernel is 
    #(double integral)\prod_{j=1}^d (1 + (w_j)/3)
    #(single integral)\prod_{j=1}^d (1 + (w_j)(1-x^2)/2)
    #(kernel) \prod_{j=1}^d (1 + w_j(1 - max(x_j, t_j)))
    if callable(method):      #discrepancy function given by the user
        #The function given by the discrepancy function must return the double integral, second integral, and kernel in this order
        double_integral, single_integral, kernel = method(x)
        #returns the discrepancy
        return np.sqrt(double_integral - (2*np.mean(single_integral)) + np.mean(np.mean(kernel)))
    else:
        #X_expanded = np.zeros((n,n,d)) + x      #Copies x into a 3d matrix of size n by n by d.
        X_expanded = np.resize(x, (1, n, d))
        Y = np.resize(x, (n, 1, d))             #reshapes x so that we can iteratively find the value of the kernels
        if method.lower() == "s" or method.lower() == "star":           #Star
            double_integral = (4/3)**d
            single_integral = ((3-x**2)/2).prod(axis=1)
            kernel = (2 - max(X_expanded, Y)).prod(axis=2)
        elif method.lower() == "l2" or method.lower() == "l2star":        #L2star
            double_integral = (1/3)**d
            single_integral = ((1-x**2)/2).prod(axis=1)
            kernel = (1 - np.maximum(X_expanded, Y)).prod(axis=2)
        elif method.lower() == "c" or method.lower() == "centered":         #Centered
            double_integral = (13/12)**d
            single_integral = (1 + (.5*abs(x - .5)) - (.5*((x -.5)**2))).prod(axis=1)
            kernel = (1 + (.5*abs(X_expanded - .5)) + (.5*abs(Y - .5)) - (.5*abs(X_expanded - Y))).prod(axis=2)
        elif method.lower() == "sy" or method.lower() == "symmetric":        #Symmetric
            double_integral = (4/3)**d
            single_integral = (1 + 2*x - (2*(x**2))).prod(axis=1)
            kernel = (2 - (2*abs(X_expanded - Y))).prod(axis=2)
        elif method.lower() == "wa" or method.lower() == "wrap around" or method.lower() == "wrap-around":        #Wrap around
            double_integral = -(4/3)**d
            single_integral = 0
            kernel = (1.5 - (abs(X_expanded - Y)*(1 - abs(X_expanded - Y)))).prod(axis=2)
        else:
            return False
    #returns the discrepancy
    return np.sqrt(double_integral - (2*np.mean(single_integral)) + np.mean(np.mean(kernel)))
