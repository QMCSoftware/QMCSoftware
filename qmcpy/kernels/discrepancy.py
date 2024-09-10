import numpy as np
import time
from copy import copy
from inspect import signature

#Make a function for double integral, single integral, and kernel
#Cut down on the copies for x
#Make notes throughout the code
#Make a plot of runtimes based on different values of n and limiter     

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
        elif method.lower() == "s" or method.lower() == "star":        #L2star
            double_integral = (1/3)**d
            single_integral = ((1-x**2)/2).prod(axis=1)
            kernel = (1 - np.maximum(X_expanded, Y)).prod(axis=2)
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

def discrepancy2(x, method = None, double_integral = None, single_integral = None, kernel = None, weight = 1, limiter = 2**25, Time = False):
    if Time == True:                #Times the actual calculation for discrepancy
        start_time = time.time()

    n, d = x.shape  #Finds the number of samples and their dimensions

    weight = weight * np.ones(d) #Makes sure that the weight is a vector, but the user can use a scalar value

    if method != None:
        if type(method) is str:
            if method.lower() == "l2" or method.lower() == "l2star":           #Star
                double_integral = lambda w : (1 + (w/3)).prod(axis=0)
                single_integral = lambda x, w : ((1 + (w*(1 - x**2)/2))).prod(axis=1)
                kernel = lambda x, y, w : (1 + w*(1 - np.maximum(x, y))).prod(axis=2)
            elif method.lower() == "s" or method.lower() == "star":        #L2star
                double_integral = lambda d: (1/3)**d
                single_integral = lambda x, w : ((1-x**2)/2).prod(axis=1)
                kernel = lambda x, y, w : (1 - np.maximum(x, y)).prod(axis=2)
            elif method.lower() == "c" or method.lower() == "centered" or method.lower() == 'cd':         #Centered
                double_integral = lambda w : (1 + (w/12)).prod()
                single_integral = lambda x, w : (1 + (0.5*w)*(abs(x - .5)*(1 - abs(x -.5)))).prod(axis=1)
                kernel = lambda x, y, w : (1 + (0.5*w)*(abs(x - .5) + abs(y - .5) - abs(x - y))).prod(axis=2)
            elif method.lower() == "sy" or method.lower() == "symmetric":        #Symmetric
                double_integral = lambda w : (1 +(w/3)).prod(axis=0)
                single_integral = lambda x, w: (1 + (w*2*x) - (w*2*(x**2))).prod(axis=1)
                kernel = lambda x, y, w : (1 + w*(1 - abs(x - y))).prod(axis=2)
            elif method.lower() == "wa" or method.lower() == "wrap around" or method.lower() == "wrap-around" or method.lower() == 'wd':        #Wrap around
                double_integral = lambda w : -(1 + (w/3)).prod(axis=0)
                single_integral = lambda x, w: 0
                kernel = lambda x, y, w: (1.5 - (abs(x - y)*(1 - abs(x - y)))).prod(axis=2)
                #double_integral = -(4/3)**d
                #single_integral
            elif method.lower() == "m" or method.lower() == "mixture" or method.lower() == 'md':        #Wrap around
                double_integral = lambda d: (19/12)**d
                single_integral = lambda x, w: ((5/3) - (.25*abs(x - .5)) - (.25*((x -.5)**2))).prod(axis=1)
                kernel = lambda x, y, w: (1.875 - (.25*abs(x - .5)) - (.25*abs(y - .5)) - (.75*abs(x - y)) + (.5*((x - y)**2))).prod(axis=2)
    #Calculates the double integral which requires no for loops
    if weight == np.ones(d):
        DI = double_integral(d)
    else:
        DI = double_integral(weight)

    single_integral_sum = 0
    kernel_sum = 0
    if x[0,0] != None:  #the discrepancy of some points
        len_chunk = np.floor(np.sqrt(limiter/d)).astype('int')
        n_chunk = np.ceil(n/len_chunk).astype('int')
        for ii in range(n_chunk):
            n_ii_start = ii*len_chunk
            n_ii_end = np.array([n,(ii+1)*len_chunk]).min()
            n_ii_batch = n_ii_end - n_ii_start
            if n_ii_batch > 0:
                x_chunk = x[n_ii_start:n_ii_end,:]
                single_integral_sum  += single_integral(x_chunk,weight).sum(axis = 0)
                x_chunk = x_chunk.reshape(n_ii_batch,1,d)
                y_chunk = x_chunk.reshape(1,n_ii_batch,d)
                kernel_sum += kernel(x_chunk,y_chunk,weight).sum()
                for jj in range(ii+1,n_chunk):
                    n_jj_start = jj*len_chunk
                    n_jj_end = np.array([n,(jj+1)*len_chunk]).min()
                    n_jj_batch = n_jj_end - n_jj_start
                    if n_jj_batch > 0:
                        y_chunk = x[n_jj_start:n_jj_end,:].reshape(1,n_jj_batch,d)
                        kernel_sum += 2*kernel(x_chunk,y_chunk,weight).sum()
        single_integral_sum  *= (2/n)
        kernel_sum *= (1/(n**2)) 
    # print(dbl_integ,sngl_integ_sum,dbl_sum)
    out = np.sqrt(DI - single_integral_sum + kernel_sum)

    if Time:  #if we are measuring time
        total_time = time.time() - start_time
        out = [out, total_time]
    return out