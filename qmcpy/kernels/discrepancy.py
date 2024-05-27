import numpy as np
import time
from copy import copy
from inspect import signature
import math

def discrepancy(method, x, weight = 1):
    n, d = x.shape                              #Finds the number of samples and the dimensions for our x_i's
    weight = weight * np.ones(d)                #if weight is a scalar, it gets turned into an array.
                                                #if weight is an array, it would still be an array since
                                                #np.ones(d) would be identified as an identity.
    #find a way to generate the weight, and generate the scalar into an array
    #For L2 star, the weighted kernel is 
    #(double integral)\prod_{j=1}^d (1 + (w_j)/3)
    #(single integral)\prod_{j=1}^d (1 + (w_j)(1-x^2)/2)
    #(kernel) \prod_{j=1}^d (1 + w_j(1 - max(x_j, t_j)))

    if callable(method):      #discrepancy function given by the user
        sig = signature(method)
        params = sig.parameters
        if len(params) == 1:
            #The function given by the discrepancy function must return the double integral, second integral, and kernel in this order
            double_integral, single_integral, kernel = method(x)
            #returns the discrepancy
        elif len(params) == 2:
            #The weighted discrepancy given by the user
            double_integral, single_integral, kernel = method(x, weight)
        return np.sqrt(double_integral - (2*np.mean(single_integral)) + np.mean(np.mean(kernel)))
    else:
        #X_expanded = np.zeros((n,n,d)) + x      #Copies x into a 3d matrix of size n by n by d.
        X_expanded = np.resize(x, (1, n, d))
        Y = np.resize(x, (n, 1, d))             #reshapes x so that we can iteratively find the value of the kernels
        if method.lower() == "l2" or method.lower() == "l2star":           #Star
            double_integral = (1 + (weight/3)).prod(axis=0)
            single_integral = ((1 + (weight*(1 - x**2)/2))).prod(axis=1)
            kernel = (1 + weight*(1 - np.maximum(X_expanded, Y))).prod(axis=2)
        #elif method.lower() == "s" or method.lower() == "star":        #L2star
        #    double_integral = (1/3)**d
        #    single_integral = ((1-x**2)/2).prod(axis=1)
        #    kernel = (1 - np.maximum(X_expanded, Y)).prod(axis=2)
        elif method.lower() == "c" or method.lower() == "centered" or method.lower() == 'cd':         #Centered
            double_integral = (13/12)**d
            single_integral = (1 + (.5*abs(x - .5)) - (.5*((x -.5)**2))).prod(axis=1)
            kernel = (1 + (.5*abs(X_expanded - .5)) + (.5*abs(Y - .5)) - (.5*abs(X_expanded - Y))).prod(axis=2)
        elif method.lower() == "sy" or method.lower() == "symmetric":        #Symmetric
            double_integral = (4/3)**d
            single_integral = (1 + 2*x - (2*(x**2))).prod(axis=1)
            kernel = (2 - (2*abs(X_expanded - Y))).prod(axis=2)
        elif method.lower() == "wa" or method.lower() == "wrap around" or method.lower() == "wrap-around" or method.lower() == 'wd':        #Wrap around
            double_integral = -(4/3)**d
            single_integral = 0
            kernel = (1.5 - (abs(X_expanded - Y)*(1 - abs(X_expanded - Y)))).prod(axis=2)
            #double_integral = -(4/3)**d
            #single_integral
        elif method.lower() == "m" or method.lower() == "mixture" or method.lower() == 'md':        #Wrap around
            double_integral = (19/12)**d
            single_integral = ((5/3) - (.25*abs(x - .5)) - (.25*((x -.5)**2))).prod(axis=1)
            kernel = (1.875 - (.25*abs(X_expanded - .5)) - (.25*abs(Y - .5)) - (.75*abs(X_expanded - Y)) + (.5*((X_expanded - Y)**2))).prod(axis=2)
        else:
            return False
    #returns the discrepancy
    return np.sqrt(double_integral - (2*np.mean(single_integral)) + np.mean(np.mean(kernel)))

def discrepancy2(method, x, weight = 1, limiter = 2*22, Time = False):
    n, d = x.shape  #initialize the list as empty

    weight = weight * np.ones(d)
    limiter = int(limiter / (2**math.ceil(math.log(d,2))))
    limiter = int(np.sqrt(4**int(math.log(limiter, 4))))

    X_expanded = np.resize(x, (1, n, d))
    Y = np.resize(x, (n, 1, d))
    iterated_X = []
    iterated_X_expanded = []
    iterated_Y = []

    for i_1 in range(int(n/limiter)+1):               #These 4 lines are used to make these lists into chunks
        iterated_X = iterated_X + [x[i_1*limiter: (i_1+1)*limiter, :]]
        iterated_X_expanded = iterated_X_expanded + [X_expanded[:, i_1*limiter: (i_1+1)*limiter, :]]
        iterated_Y = iterated_Y + [Y[i_1*limiter: (i_1+1)*limiter, :, :]]


    if n%limiter == 0:
        iterated_X = iterated_X[0: len(iterated_X) - 1]
        iterated_X_expanded = iterated_X_expanded[0:len(iterated_X_expanded)-1]
        iterated_Y = iterated_Y[0:len(iterated_Y)-1]
    
    if Time == True:
        start_time = time.time()

    if len(method) == 3:
        print(0)
    else:
        if method.lower() == "l2" or method.lower() == "l2star":           #Star
            DI = (1 + (weight/3)).prod(axis=0)

            B = 0
            for j in range(len(iterated_X)):
                x = iterated_X[j]
                single_integral = ((1 + (weight*(1 - x**2)/2))).prod(axis=1)
                B = B + np.sum(single_integral)

            SI = B*2/n

            C = 0
            for j in range(len(iterated_X_expanded)):
                for i in range(len(iterated_Y)):
                    x = iterated_X_expanded[j]
                    y = iterated_Y[i]
                    kernel = (1 + weight*(1 - np.maximum(x, y))).prod(axis=2)
                    C = C + np.sum(np.sum(kernel))

            K = C/(n**2)
        elif method.lower() == "c" or method.lower() == "centered" or method.lower() == 'cd':
            DI = (13/12)**d

            B = 0
            for j in range(len(iterated_X)):
                x = iterated_X[j]
                single_integral = (1 + (.5*abs(x - .5)) - (.5*((x -.5)**2))).prod(axis=1)
                B = B + np.sum(single_integral)

            SI = B*2/n

            C = 0
            for j in range(len(iterated_X_expanded)):
                for i in range(len(iterated_Y)):
                    x = iterated_X_expanded[j]
                    y = iterated_Y[i]
                    kernel = (1 + (.5*abs(x - .5)) + (.5*abs(y - .5)) - (.5*abs(x - y))).prod(axis=2)
                    C = C + np.sum(np.sum(kernel))

            K = C/(n**2)
        elif method.lower() == "sy" or method.lower() == "symmetric":        #Symmetric
            DI = (4/3)**d

            B = 0
            for j in range(len(iterated_X)):
                x = iterated_X[j]
                single_integral = (1 + 2*x - (2*(x**2))).prod(axis=1)
                B = B + np.sum(single_integral)

            SI = B*2/n

            C = 0
            for j in range(len(iterated_X_expanded)):
                for i in range(len(iterated_Y)):
                    x = iterated_X_expanded[j]
                    y = iterated_Y[i]
                    kernel = (2 - (2*abs(x - y))).prod(axis=2)
                    C = C + np.sum(np.sum(kernel))

            K = C/(n**2)
        elif method.lower() == "wa" or method.lower() == "wrap around" or method.lower() == "wrap-around" or method.lower() == 'wd':        #Wrap around
            DI = -(4/3)**d

            SI = 0

            C = 0
            for j in range(len(iterated_X_expanded)):
                for i in range(len(iterated_Y)):
                    x = iterated_X_expanded[j]
                    y = iterated_Y[i]
                    kernel = (1.5 - (abs(x - y)*(1 - abs(x - y)))).prod(axis=2)
                    C = C + np.sum(np.sum(kernel))

            K = C/(n**2)
        elif method.lower() == "m" or method.lower() == "mixture" or method.lower() == 'md':        #Wrap around
            DI = (19/12)**d

            B = 0
            for j in range(len(iterated_X)):
                x = iterated_X[j]
                single_integral = ((5/3) - (.25*abs(x - .5)) - (.25*((x -.5)**2))).prod(axis=1)
                B = B + np.sum(single_integral)

            SI = B*2/n

            C = 0
            for j in range(len(iterated_X_expanded)):
                for i in range(len(iterated_Y)):
                    x = iterated_X_expanded[j]
                    y = iterated_Y[i]
                    kernel = (1.875 - (.25*abs(x - .5)) - (.25*abs(y - .5)) - (.75*abs(x - y)) + (.5*((x - y)**2))).prod(axis=2)
                    C = C + np.sum(np.sum(kernel))

            K = C/(n**2)
        else:
            return False
        #returns the discrepancy
    if Time == True:
        total_time = time.time() - start_time
        return np.sqrt(DI - SI + K), total_time
    if Time == False:
        print(DI)
        print(SI)
        print(K)
        return np.sqrt(DI - SI + K)