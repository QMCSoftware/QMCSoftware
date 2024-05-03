import numpy as np
import time
from copy import copy
from inspect import signature
import math

def discrepancy(method, x, weight = 1, limiter = 2**22):
    n, d = x.shape                              #Finds the number of samples and the dimensions for our x_i's
    weight = weight * np.ones(d)                #if weight is a scalar, it gets turned into an array.
                                                #if weight is an array, it would still be an array since
                                                #np.ones(d) would be identified as an identity.
    #find a way to generate the weight, and generate the scalar into an array
    #For L2 star, the weighted kernel is 
    #(double integral)\prod_{j=1}^d (1 + (w_j)/3)
    #(single integral)\prod_{j=1}^d (1 + (w_j)(1-x^2)/2)
    #(kernel) \prod_{j=1}^d (1 + w_j(1 - max(x_j, t_j)))

    sample_size = int(limiter / d)
    steps = int(n/sample_size)
    if d != 1:
        limiter = limiter / (2**int(math.log(d-1,2)+1))

    limiter = int((math.log(limiter-1, 2) +1)/2)

    X = []                                          #initialize the list as empty
    for i in range(int(n/limiter)+1):               #These 2 lines seperate the samples into chunks
        X = X + [x[i*limiter: (i+1)*limiter, :]]
    if X[-1] == []:
        X = X[0: len(X) - 1]
    
    X_expanded = []
    Y = []
    for i in range(int(n/limiter)+1):
        A = X[i*limiter:(i+1)*limiter, :]
        c, f = A.shape
        X_expanded = X_expanded + [np.resize(A, (1, c, f))]
        Y = Y + [np.resize(A, (c, 1, f))]

    if X_expanded[-1] == []:
        X_expanded = X_expanded[0:len(X_expanded) -1]

    if Y[-1] == []:
        Y = Y[0:len(Y)-1]
    

    if len(method) == 3:      #discrepancy function given by the user
        #Start with the double integral function
        sig_1 = signature(method[0]) #This line and the line below figures out how many variables the function takes
        params_1 = sig_1.parameters
        if len(params_1) == 1:    #Not weighted
            #If it is not weighted, go ahead and find the double integral with $d$ being your input
            double_integral = method[0](d)
        
        elif len(params_1) == 2:  #Weighted
            #The user must have d first and then 'w' the weight
            double_integral = method[0](d, weight)

        #Now this is single integral
        sig_2 = signature(method[1])
        params_2 = sig_2.parameters
        total_single_integral = 0
        if len(params_2) == 1: #Not weighted
            #If it is not weighted, go ahead and compute the first integral
            for i in range(len(X)):   
                single_integral = method[1](X[i])
                total_single_integral += np.sum(single_integral)

        elif len(params_2) == 2: #Weighted
            #If it is weighted, first variable x, second variable 'w' which is the weight
            for i in range(len(X)):   
                single_integral = method[1](X[i], weight)
                total_single_integral += np.sum(single_integral)

        sig_3 = signature(method[2])
        params_3 = sig_3.parameters
        total_kernel = 0
        if len(params_3) == 2: #Not weighted
            #If it is not weighted, go ahead and compute the kernels
            for i in range(len(X_expanded)):   
                for j in range(len(Y)):
                    kernel = method[2](X_expanded[i], Y[j])
                    total_kernel += np.sum(np.sum(kernel))

        elif len(params_3) == 3: #Weighted
            #If it is weighted, first variable x, second variable 'w' which is the weight
            for i in range(len(X_expanded)):   
                for j in range(len(Y)):
                    kernel = method[2](X_expanded[i], Y[j], weight)
                    total_kernel += np.sum(np.sum(kernel))

        return np.sqrt(double_integral - (2*(total_single_integral)/n) + (total_kernel/ (n**2)))
    else:
        #X_expanded = np.zeros((n,n,d)) + x      #Copies x into a 3d matrix of size n by n by d.
        X_expanded = np.resize(x, (1, n, d))
        Y = np.resize(x, (n, 1, d))             #reshapes x so that we can iteratively find the value of the kernels
        if method.lower() == "l2" or method.lower() == "l2star":           #Star
            double_integral = (1 + (weight/3)).prod(axis=0)
            single_integral_total = 0
            for sample in range(steps):
                single_integral = ((1 + (weight*(1 - x[sample*sample_size:(sample+1)*sample_size, :]**2)/2))).prod(axis=1)
                single_integral_total = single_integral_total + np.sum(single_integral)
            kernel_total = 0
            for sample_1 in range(steps):
                for sample_2 in range(steps):
                    kernel = (1 + weight*(1 - np.maximum(X_expanded[sample_1*sample_size: (sample_1+1)*sample_size, :], Y[sample_2*sample_size: (sample_2+1)*sample_size, :]))).prod(axis=2)
                    kernel_total = kernel_total + np.sum(np.sum(kernel))
            return np.sqrt(double_integral - 2*(single_integral_total/n) + (kernel_total / n**2))
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

