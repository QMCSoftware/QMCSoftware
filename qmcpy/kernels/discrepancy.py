import numpy as np
import time
from copy import copy
from inspect import signature   

#Get rid of the double_integral, single_integral, and kernels and make method into an array
#Go back into the Scipy and replace the old discrepancy
#Make a demo in the jupyter notebook
#Unit tests
#And then later a blog
def discrepancy(x, method, weight = 1, limiter = 2**25, Time = False):
    if Time == True:                #Times the actual calculation for discrepancy
        start_time = time.time()

    n, d = x.shape  #Finds the number of samples and their dimensions

    #reconfigures the weight so that it is appropriate to the given matrix
    if type(weight) == list: # if weight is a list
        weight = weight[0:d] #make sure you take the first d elements for calculations
    else:
        weight = weight * np.ones(d) #if weight is scalar, just make a list of d weights.

    if type(method) == str:    #If a method was chosen
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
                single_integral = 0
                kernel = lambda x, y, w: (1.5 - (abs(x - y)*(1 - abs(x - y)))).prod(axis=2)
            elif method.lower() == "m" or method.lower() == "mixture" or method.lower() == 'md':        #Wrap around
                double_integral = lambda w: (((7/12)*w)+1).prod()
                single_integral = lambda x, w: (1 + w*((2/3) - (.25*abs(x - .5)) - (.25*((x -.5)**2)))).prod(axis=1)
                kernel = lambda x, y, w: (1+ w*(.875 - (.25*abs(x - .5)) - (.25*abs(y - .5)) - (.75*abs(x - y)) + (.5*((x - y)**2)))).prod(axis=2)
    else:
        #If the user punches in a list of functions for variable method
        if type(method) == list:
            if len(method) == 2:
                #if we have 2 functions, define the double integral and kernel. And set single_integral to 0.
                double_integral = method[0]
                single_integral = 0     
                kernel = method[1]
            elif len(method) == 3:
                double_integral = method[0]
                single_integral = method[1]
                kernel = method[2]
    # Calculates the double integral which doesn't require loops
    if len(signature(kernel).parameters) == 2:
        DI = double_integral(d)
    else:
        DI = double_integral(weight)

    # initializing the sum of the single integrals
    single_integral_sum = 0
    # initializing the sum of kernels
    kernel_sum = 0

    #cutting down on run time depending on if single_integral is 0 or not.
    if single_integral != 0:
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
                    if len(signature(double_integral).parameters) == 1:
                        single_integral_sum  += single_integral(x_chunk).sum(axis = 0)
                    else:
                        single_integral_sum  += single_integral(x_chunk,weight).sum(axis = 0)
                    # We have to reshape the matrix x such that we get an iteration.
                    x_chunk = x_chunk.reshape(n_ii_batch,1,d)
                    y_chunk = x_chunk.reshape(1,n_ii_batch,d)
                    #Calculates the kernel sum for the given chunk since they are the same
                    if len(signature(double_integral).parameters) == 2:
                        kernel_sum += kernel(x_chunk,y_chunk).sum()
                    else:
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
                            if len(signature(double_integral).parameters) == 2:
                                kernel_sum += 2*kernel(x_chunk,y_chunk).sum()
                            else:
                                kernel_sum += 2*kernel(x_chunk,y_chunk,weight).sum()
    else:
        #The only difference is that we are no longer calculating for single integral, since it is 0.

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
                    # We don't need single integral since it is equal to 0

                    # We have to reshape the matrix x such that we get an iteration.
                    x_chunk = x_chunk.reshape(n_ii_batch,1,d)
                    y_chunk = x_chunk.reshape(1,n_ii_batch,d)
                    #Calculates the kernel sum for the given chunk since they are the same
                    if len(signature(double_integral).parameters) == 2:
                        kernel_sum += kernel(x_chunk,y_chunk).sum()
                    else:
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
                            if len(signature(double_integral).parameters) == 2:
                                kernel_sum += 2*kernel(x_chunk,y_chunk).sum()
                            else:
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