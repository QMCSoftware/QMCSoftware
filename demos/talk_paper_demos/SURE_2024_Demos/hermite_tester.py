import matplotlib.pyplot as plt
import math
import numpy as np
import scipy as sp
import qmcpy as qp
from scipy import stats
from copy import deepcopy
import sympy as sy
import sympy.stats
import pprofile
import sys

# QMC Point Generators
def gen_iid_ld_pts(dimen = 3, n = 2**12):
    """
    Generate IID, Sobol, Lattic, DNB2, and Halton sample points,
    based on dimension and smaple size.
    """
    coord_wts = 2.0**(-np.array(range(0, dimen)))
    iidpts = qp.IIDStdUniform(dimen).gen_samples(n)
    sobpts = qp.Sobol(dimen).gen_samples(n)
    latticepts = qp.Lattice(dimen).gen_samples(n)
    dnetb2pts = qp.DigitalNetB2(dimen).gen_samples(n)
    haltonpts = qp.Halton(dimen).gen_samples(n)
    return (iidpts, sobpts, latticepts, dnetb2pts, haltonpts)

# Kernels
# hardcoding hermite kernel coefficients
def hardcode_hermite_kernel(r):
    '''
    returns hermite coefficients for r = 0, 1, or 2
    band_width = the kernel band width, often denoted h; a smaller bandwidth means a peakier kernel
    r = number of terms in our kernel
    '''
    if r == 0:
        return sy.Matrix([[1/(math.pi)**0.5]])
    elif r == 1:
        return sy.Matrix([[1/(math.pi)**0.5, -1/(4*(math.pi)**0.5)]])
    elif r == 2:
        return sy.Matrix([[1/(math.pi)**0.5, -1/(4*(math.pi)**0.5), 1/(32*(math.pi)**0.5)]])
    else:
        raise ValueError("Invalid input. r must be 0, 1, or 2.")

# hardcoded: define hermite kernel for r = 0 (corresponds to r = 1 in the original case)
def hardcode_hermite_kernel_weight(y, r = 0):
    '''
    r = number of terms in our kernel
    '''
    coef = np.array((hardcode_hermite_kernel(r = r)))
    # print(coef)
    # poly = sp.special.hermite(0)
    # print(poly)
    k = sp.special.hermite(0)(y) * coef[0][0] #initialize a vector of kernel values
    # print(k)
    for ii in range(0, r):
        k += sp.special.hermite(2*ii)(y) * coef[0][ii] #add the additional terms
    k *= np.exp(-y*y/2) #normalizing weight for Hermite functions #form the isotropic kernel and insert the bandwidth dependency
    return k

# PDFs
# Unamed distribution
def testfun(x):
    return 10 * np.exp(-x) * np.sin(np.pi*x)

# Guassian distribution
def guass_distr(x, coord_wts = 1): # function f(x) defines the random variable
    wtx = np.multiply(x, coord_wts)
    y = 10 * np.exp(-wtx.sum(1)) * np.sin(np.pi * wtx.sum(1))
    #y = x[:,0] # if x is uniform, then y is uniform
    return y

# KDEs
def kde_pt(kernel, ypts, bandwidth, yeval):
    return np.mean(kernel((yeval-ypts) / bandwidth)) / bandwidth

# def estimated_pts(x):
#     return kde_pt(hardcode_hermite_kernel_weight, pts, h, x)

# def squared_difference_pts(x):
#     return (hardcode_hermite_kernel_weight - estimated_pts(x))**2

# Integration operation from SciPy
def integration_test(h, pts, kde, kernel, true_function):
        def estimated_pts(x):
            return kde(kernel, pts, h, x)
        def squared_difference_pts(x):
            return (true_function(x) - estimated_pts(x))**2
        mise, error = sp.integrate.quad(squared_difference_pts, -4, 8) # This seems to be taking the longest. Will investigate.
        return (mise, error)

# All operations
def optimal_h(h_space, pts, kde, kernel, true_function):
    '''
    returns optimal bandwidth and lowest value of MISE
    h_space: a set of bandwidths to be tested (e.g., np.linspace(0.001, 3, 100))
    pts: set of randomly generated points (IID or LD)
    kde: kde function
    kernel: function that defines kernel
    true_function: function that we are trying to estimate (e.g., f(x), laplace, uniform, exponential, etc.)
    '''
    h_optimal = 0
    lowest_mise = 100000
    for h in h_space:
        def estimated_pts(x): # TODO: Move out of for-loop
            return kde(kernel, pts, h, x)
        def squared_difference_pts(x):
            return (true_function(x) - estimated_pts(x))**2
        mise, error = sp.integrate.quad(squared_difference_pts, -4, 8) # TODO: Fix this interval of integration so that it makes sense. Should probably be parameterized as well.
        if mise < lowest_mise:
            lowest_mise = mise
            h_optimal = h

    return(lowest_mise, h_optimal)

# Functions to test
# Alternatives to SciPy's integration

if __name__ == '__main__':
    iidpts1, sobpts1, latticepts1, dnetb2pts1, haltonpts1 = gen_iid_ld_pts(dimen = 1, n = 2**12)
    yiid1 = guass_distr(iidpts1)
    test_option = sys.argv[1]

    # TODO: Add option for CProfiler
    profiler = pprofile.Profile()
    with profiler:
        if test_option in ('optimal_h' ,'0'):
            result = optimal_h(np.linspace(0.001, 2, 2), yiid1, kde_pt, hardcode_hermite_kernel_weight, testfun)
        elif test_option in ('kde_pt', '1'):
            result = kde_pt(hardcode_hermite_kernel_weight, yiid1, 0.5, 2.0)
        elif test_option in ('hardcode_hermite_kernel', '2'):
            r_number = int(sys.argv[2])
            hardcode_hermite_kernel(r_number)
        elif test_option in ('hardcode_hermite_kernel', '3'):
            r_number = int(sys.argv[2])
            result = hardcode_hermite_kernel_weight(y=yiid1, r=r_number)
        elif test_option in ('integration_test', '4'):
            mise, error = integration_test(0.1, yiid1, kde_pt, hardcode_hermite_kernel_weight, testfun)
        else:
            raise Exception

    profiler.print_stats()