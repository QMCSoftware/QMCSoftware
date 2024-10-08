import numpy as np
import time
from copy import copy
from inspect import signature   

def discrepancy(method, x, weight = 1):
    n, d = x.shape                              #Finds the number of samples and the dimensions for our x_i's
    weight = weight * np.ones(d)                #if weight is a scalar, it gets turned into an array.
                                                #if weight is an array, it would still be an array since
                                                #np.ones(d) would be identified as an identity.
    #find a way to generate the weight, and generate the scalar into an array

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


    #reconfigures the weight so that it is appropriate to the given matrix
    if type(weight) == list: # if weight is a list
        weight = weight[0:d] #make sure you take the first d elements for calculations
    else:
        weight = weight * np.ones(d) #if weight is scalar, just make a list of d weights.

    if method != None:    #If a method was chosen
        if type(method) is str: #Give the double and single integral along with the kernel based on the method
            if method.lower() == "l2" or method.lower() == "l2star":           #Star
                double_integral = lambda w : (1 + (w/3)).prod(axis=0)
                single_integral = lambda x, w : ((1 + (w*(1 - x**2)/2))).prod(axis=1)
                kernel = lambda x, y, w : (1 + w*(1 - np.maximum(x, y))).prod(axis=2)
            #elif method.lower() == "s" or method.lower() == "star":        #L2star
            #    double_integral = lambda d: (1/3)**d
            #    single_integral = lambda x, w : ((1-x**2)/2).prod(axis=1)
            #    kernel = lambda x, y, w : (1 - np.maximum(x, y)).prod(axis=2)
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
            elif method.lower() == "m" or method.lower() == "mixture" or method.lower() == 'md':        #Wrap around
                double_integral = lambda w: (((7/12)*w)+1).prod()
                single_integral = lambda x, w: (1 + w*((2/3) - (.25*abs(x - .5)) - (.25*((x -.5)**2)))).prod(axis=1)
                kernel = lambda x, y, w: (1+ w*(.875 - (.25*abs(x - .5)) - (.25*abs(y - .5)) - (.75*abs(x - y)) + (.5*((x - y)**2)))).prod(axis=2)
    # Calculates the double integral which doesn't require loops
    DI = double_integral(weight)

    # initializing the sum of the single integrals
    single_integral_sum = 0
    # initializing the sum of kernels
    kernel_sum = 0

    # As long as x is a matrix
    if x[0,0] != None:  #the discrepancy of some points
        # These 2 rows give out the limit as to how many samples to consider when calculating
        # the sum of single integral and kernels, so that the computer does not get overwhelmed
        # with the calculations
        len_chunk = np.floor(np.sqrt(limiter/d)).astype('int')
        n_chunk = np.ceil(n/len_chunk).astype('int')
        for ii in range(n_chunk):
            # Gets a chunk from the matrix x
            n_ii_start = ii*len_chunk
            n_ii_end = np.array([n,(ii+1)*len_chunk]).min()
            n_ii_batch = n_ii_end - n_ii_start
            #As long as the n_ii_start and n_ii_end are not the same
            if n_ii_batch > 0:
                # Grab some samples from x
                x_chunk = x[n_ii_start:n_ii_end,:]
                # Go ahead and calculate the single_integral for those samples and add them up
                # to single_integral_sum
                single_integral_sum  += single_integral(x_chunk,weight).sum(axis = 0)
                # We have to reshape the matrix x such that we get an iteration.
                x_chunk = x_chunk.reshape(n_ii_batch,1,d)
                y_chunk = x_chunk.reshape(1,n_ii_batch,d)
                #Calculates the kernel sum for the given chunk since they are the same
                kernel_sum += kernel(x_chunk,y_chunk,weight).sum()
                for jj in range(ii+1,n_chunk):
                    #This is the part where you need an iteration for the rest of the kernels for when 
                    n_jj_start = jj*len_chunk
                    n_jj_end = np.array([n,(jj+1)*len_chunk]).min()
                    n_jj_batch = n_jj_end - n_jj_start
                    if n_jj_batch > 0:
                        y_chunk = x[n_jj_start:n_jj_end,:].reshape(1,n_jj_batch,d)
                        # We multiply by 2, because you would have to do the same calculation twice, sp
                        # there was no reason to repeat. Multiply the kernel when x_chunk != y_chunk by
                        # 2.
                        kernel_sum += 2*kernel(x_chunk,y_chunk,weight).sum()
        # Take the average of both the single integral and the kernel
        single_integral_sum  *= (2/n)
        kernel_sum *= (1/(n**2)) 
    # Calculates the discrepancy
    out = np.sqrt(DI - single_integral_sum + kernel_sum)

    if Time:  #if we are measuring time
        total_time = time.time() - start_time
        out = [out, total_time]
    return out