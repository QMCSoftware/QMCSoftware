import numpy as np
import time
from copy import copy
from inspect import signature

#Make a function for double integral, single integral, and kernel
#Cut down on the copies for x
#Make notes throughout the code
#Make a plot of runtimes based on different values of n and limiter

def double_integral(method, weight, d):
    if callable(method):            #discrepancy function given by the user
        sig = signature(method)     #[These 2 lines, were set up to figure out how many variables the function has
        params = sig.parameters     #[
        if len(params) == 1:
            #if method function from user has 1 variable, it must be a function of d
            return method(d)
        elif len(params) == 2:
            #otherwise, for a weighted double integral, you need method function to
            #have 2 variables, so that the code would know that it is weighted
            return method(d, weight)
    else:
        if method.lower() == "l2" or method.lower() == "l2star":
            # ^^^ function for l2star double integral
            return (1 + (weight/3)).prod(axis=0)
        elif method.lower() == "c" or method.lower() == "centered" or method.lower() == 'cd':
            # ^^^ function for centered double integral
            return (weight*(13/12)).prod(axis=0)
        elif method.lower() == "sy" or method.lower() == "symmetric":
            # ^^^ function for symmetric double integral
            return (weight*(4/3)).prod(axis=0)
        elif method.lower() == "wa" or method.lower() == "wrap around" or method.lower() == "wrap-around" or method.lower() == 'wd':        #Wrap around
            # ^^^ function for wrap around double integral
            return -(weight * 4/3).prod(axis=0)
        elif method.lower() == "m" or method.lower() == "mixture" or method.lower() == 'md':
            # ^^^ function for mixture double integral
            return (weight*(19/12)).prod(axis=0)
        elif type(method) is float or type(method) is int:
            # ^^^ incase if user puts in fixed value.
            return method
        else:
            #If all ends fail, return False. We can work on an error system later
            return False
        
def single_integral(method, x, weight):
    if callable(method):            #discrepancy function given by the user
        sig = signature(method)     #[These 2 lines help figure out how many variables method function has
        params = sig.parameters     #[
        if len(params) == 1:
            #Unweighted discrepancy function from the user must have 1 variable and it's the sample
            return method(x)
        elif len(params) == 2:
            #Weighted discrepancy function from the user must have 2 variables, 'x' and 'weight' in the exact order
            return method(x, weight)
    else:
        if method.lower() == "l2" or method.lower() == "l2star":
            # ^^^ function for l2star single integral
            return ((1 + (weight*(1 - x**2)/2))).prod(axis=1)
        elif method.lower() == "c" or method.lower() == "centered" or method.lower() == 'cd':
            # ^^^ function for centered single integral
            return (1 + (.5*abs(x - .5)) - (.5*((x -.5)**2))).prod(axis=1)
        elif method.lower() == "sy" or method.lower() == "symmetric":        #Symmetric
            # ^^^ function for symmetric single integral
            return (1 + 2*x - (2*(x**2))).prod(axis=1)
        elif method.lower() == "wa" or method.lower() == "wrap around" or method.lower() == "wrap-around" or method.lower() == 'wd':        #Wrap around
            # ^^^ function for wrap around single integral
            return 0
        elif method.lower() == "m" or method.lower() == "mixture" or method.lower() == 'md':
            # ^^^ function for mixture single integral
            return ((5/3) - (.25*abs(x - .5)) - (.25*((x -.5)**2))).prod(axis=1)
        elif type(method) is float or type(method) is int:
            #Allow the user to put in a fixed value
            return method
        else:
            #We can work on an error system later
            return False
    
def kernel(method, weight, x, y):
    if callable(method):            #discrepancy function given by the user
        sig = signature(method)     #[These 2 lines is to figure out how many variables the method function contains
        params = sig.parameters     #[
        if len(params) == 2:
            #Unweighted method function has 2 variables, x and y
            return method(x,y)
        elif len(params) == 3:
            #Otherwise, if 3 then it is a weighted function and you have x and y, then weight
            return method(x,y,weight)
    else:
        if method.lower() == "l2" or method.lower() == "l2star":
            # ^^^ function for l2star kernel
            return (1 + weight*(1 - np.maximum(x, y))).prod(axis=2)
        elif method.lower() == "c" or method.lower() == "centered" or method.lower() == 'cd':
            # ^^^ function for centered kernel
            return (1 + (.5*abs(x - .5)) + (.5*abs(y - .5)) - (.5*abs(x - y))).prod(axis=2)
        elif method.lower() == "sy" or method.lower() == "symmetric":        #Symmetric
            # ^^^ function for symmetric kernel
            return (2 - (2*abs(x - y))).prod(axis=2)
        elif method.lower() == "wa" or method.lower() == "wrap around" or method.lower() == "wrap-around" or method.lower() == 'wd':        #Wrap around
            # ^^^ function for wrap around kernel
            return (1.5 - (abs(x - y)*(1 - abs(x - y)))).prod(axis=2)
        elif method.lower() == "m" or method.lower() == "mixture" or method.lower() == 'md':
            # ^^^ function for mixture kernel
            return (1.875 - (.25*abs(x - .5)) - (.25*abs(y - .5)) - (.75*abs(x - y)) + (.5*((x - y)**2))).prod(axis=2)
        elif type(method) is float or type(method) is int:
            #The user can punch in a fixed value
            return method
        else:
            #Work on error system later
            return False
        

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

def discrepancy2(method, x, weight = 1, limiter = 2**16, Time = False):
    if Time == True:                #Times the actual calculation for discrepancy
        start_time = time.time()

    n, d = x.shape  #Finds the number of samples and their dimensions

    weight = weight * np.ones(d) #Makes sure that the weight is a vector, but the user can use a scalar value

    limiter = int(limiter / d)      #Figures out how many samples the code should take in
    limiter = int(np.sqrt(limiter))

    #Gets 2 new matrix to calculate kernel
    X_expanded = np.resize(x, (1, n, d))
    Y = np.resize(x, (n, 1, d))

    #Goes ahead and starts 3 variables for calculating single integral and kernel
    #and initialize the list
    iterated_X = []
    iterated_X_expanded = []
    iterated_Y = []
    n_chunks = int(np.ceil(n/limiter))
    for i_1 in range(int(n/limiter)+1):               #These 4 lines are used to make these lists into chunks
        iterated_X = iterated_X + [x[i_1*limiter: (i_1+1)*limiter, :]]
        iterated_X_expanded = iterated_X_expanded + [X_expanded[:, i_1*limiter: (i_1+1)*limiter, :]]
        iterated_Y = iterated_Y + [Y[i_1*limiter: (i_1+1)*limiter, :, :]]

    #Gets rid of the null matrices if it does meet the criterion
    if n%limiter == 0:
        iterated_X = iterated_X[0: len(iterated_X) - 1]
        iterated_X_expanded = iterated_X_expanded[0:len(iterated_X_expanded)-1]
        iterated_Y = iterated_Y[0:len(iterated_Y)-1]
    
    #Figures out if method is a string. If it is then it sets method1, method2, and method3 as method given by user
    if type(method) is str:
        method1 = method
        method2 = method
        method3 = method
    else:
        #The only way this happens is if the user puts in a discrepancy function requiring 3 functions.
        method1 = method[0]
        method2 = method[1]
        method3 = method[2]
    #method1 is the double integral, method2 is the single integral, and method3 is the kernel

    #Calculates the double integral which requires no for loops
    DI = double_integral(method1, weight, d)
    
    #Calculates the single integral which would require 1 for loop
    B = 0                                                                   #initializes variable B with 0
    for j in range(n_chunks):                                        #For however many elements in iterated_X
        #B = B + np.sum(single_integral(method2, iterated_X[j], weight))     #Calculate the single integral per iteration and add
        B = B + np.sum(single_integral(method2, x[j*limiter: (j+1)*limiter, :], weight))

    SI = B*2/n #Finds the average and multiply by 2 for the single integral component for discrepancy

    #Calculates the kernels which requires 2 for loops
    C = 0                                                                                           #initializes C as 0
    for j in range(len(iterated_X_expanded)):                                                       #[For each iteration of
        for i in range(len(iterated_Y)):                                                            #[iterated_X_expanded & iterated_Y
            C = C + np.sum(np.sum(kernel(method3, weight, iterated_X_expanded[j], iterated_Y[i])))  #[Adds up all the values for kernel
    
    K = C/(n**2) #Finds the average for the kernels

    if Time == True:
        #If Time is true
        total_time = time.time() - start_time
        #return the discrepancy value and total time it took
        return np.sqrt(DI - SI + K), total_time
    else:
        #Otherwise, just print the discrepancy value.
        return np.sqrt(DI - SI + K)