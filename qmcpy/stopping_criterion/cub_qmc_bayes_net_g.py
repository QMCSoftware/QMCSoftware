from ._stopping_criterion import StoppingCriterion
from ..accumulate_data.ld_transform_bayes_data import LDTransformBayesData
from ..discrete_distribution import Sobol
from ..integrand import Keister
from ..util import MaxSamplesWarning, ParameterError, ParameterWarning, NotYetImplemented
from ..discrete_distribution.c_lib import c_lib
import ctypes
from numpy import log2, ctypeslib
import numpy as np
from time import time
import warnings


class CubBayesNetG(StoppingCriterion):
    """
    Stopping criterion for Bayesian Cubature using digital net (Sobol) sequence with guaranteed
    accuracy over a d-dimensional region to integrate within a specified generalized error
    tolerance with guarantees under Bayesian assumptions.

    >>> k = Keister(Sobol(2, seed=123456789))
    >>> sc = CubBayesNetG(k,abs_tol=.05)
    >>> solution,data = sc.integrate()
    >>> solution
    1.807...
    >>> data
    Solution: 1.8071         
    Keister (Integrand Object)
    Sobol (DiscreteDistribution Object)
        d               2^(1)
        randomize       1
        graycode        0
        seed            507576
        mimics          StdUniform
        dim0            0
    Gaussian (TrueMeasure Object)
        mean            0
        covariance      2^(-1)
        decomp_type     pca
    CubBayesNetG (StoppingCriterion Object)
        abs_tol         0.050
        rel_tol         0
        n_init          2^(8)
        n_max           2^(22)
    LDTransformBayesData (AccumulateData Object)
        n_total         256
        solution        1.807
        error_bound     0.014
        time_integrate  ...
        
    Adapted from
        https://github.com/GailGithub/GAIL_Dev/blob/master/Algorithms/IntegrationExpectation/cubBayesNet_g.m

    Reference
        [1] Sou-Cheng T. Choi, Yuhan Ding, Fred J. Hickernell, Lan Jiang, Lluis Antoni Jimenez Rugama,
        Da Li, Jagadeeswaran Rathinavel, Xin Tong, Kan Zhang, Yizhi Zhang, and Xuan Zhou,
        GAIL: Guaranteed Automatic Integration Library (Version 2.3) [MATLAB Software], 2019.
        Available from http://gailgithub.github.io/GAIL_Dev/

    Guarantee
        This algorithm attempts to calculate the integral of function f over the
        hyperbox [0,1]^d to a prescribed error tolerance tolfun:= max(abstol,reltol*| I |)
        with guaranteed confidence level, e.g., 99% when alpha=0.5%. If the
        algorithm terminates without showing any warning messages and provides
        an answer Q, then the following inequality would be satisfied:

                Pr(| Q - I | <= tolfun) = 99%

        This Bayesian cubature algorithm guarantees for integrands that are considered
        to be an instance of a gaussian process that fall in the middle of samples space spanned.
        Where The sample space is spanned by the covariance kernel parametrized by the scale
        and shape parameter inferred from the sampled values of the integrand.
        For more details on how the covariance kernels are defined and the parameters are obtained,
        please refer to the references below.
    """
    def __init__(self, integrand, abs_tol=1e-2, rel_tol=0,
                 n_init=2 ** 8, n_max=2 ** 22, alpha=0.01):
        self.parameters = ['abs_tol', 'rel_tol', 'n_init', 'n_max']
        # Set Attributes
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        m_min = log2(n_init)
        m_max = log2(n_max)
        if m_min % 1 != 0. or m_min < 5 or m_max % 1 != 0:
            warning_s = '''
                n_init and n_max must be a powers of 2.
                n_init must be >= 2^8.
                Using n_init = 2^8 and n_max=2^22.'''
            warnings.warn(warning_s, ParameterWarning)
            m_min = 8.
            m_max = 22.
        self.m_min = m_min
        self.m_max = m_max
        self.n_init = n_init  # number of samples to start with = 2^mmin
        self.n_max = n_max  # max number of samples allowed = 2^mmax
        self.alpha = alpha  # p-value, default 0.1%.
        self.order = 1  # Currently supports only order=1

        self.useGradient = False  # If true uses gradient descent in parameter search
        self.oneTheta = True  # If true use common shape parameter for all dimensions
        # else allow shape parameter vary across dimensions
        self.stop_at_tol = True  # automatic mode: stop after meeting the error tolerance
        self.arb_mean = True  # by default use zero mean algorithm
        self.errbd_type = 'MLE'  # Available options {'MLE', 'GCV', 'full'}

        # private properties
        # Full Bayes - assumes m and s^2 as hyperparameters
        # GCV - Generalized cross validation
        self.kernType = 1  # Type-1:

        self.avoid_cancel_error = True  # avoid cancellation error in stopping criterion
        self.uncert = 0  # quantile value for the error bound
        self.debug_enable = False  # enable debug prints
        self.data = None
        self.fwht = FWHT()

        # QMCPy Objs
        self.integrand = integrand
        self.true_measure = self.integrand.true_measure
        self.discrete_distrib = self.integrand.discrete_distrib

        # Verify Compliant Construction
        allowed_levels = ['single']
        allowed_distribs = ["Sobol"]
        super(CubBayesNetG, self).__init__(allowed_levels, allowed_distribs)

        if self.discrete_distrib.randomize == False:
            raise ParameterError("CubBayesNet_g requires discrete_distrib to have randomize=True")

    # computes the integral
    def integrate(self):
        # Construct AccumulateData Object to House Integration data
        self.data = LDTransformBayesData(self, self.integrand, self.true_measure, self.discrete_distrib, 
            self.m_min, self.m_max, self._fwht_h, self._merge_fwht, self.kernel)
        tstart = time()  # start the timer

        # Iteratively find the number of points required for the cubature to meet
        # the error threshold
        while True:
            # Update function values
            xun_, ftilde_, m = self.data.update_data()
            stop_flag, muhat, order_, err_bnd = self.data.stopping_criterion(xun_, ftilde_, m)

            # if stop_at_tol true, exit the loop
            # else, run for for all 'n' values.
            # Used to compute error values for 'n' vs error plotting
            if self.stop_at_tol and stop_flag:
                break

            if m >= self.m_max:
                warnings.warn(f'Already used maximum allowed sample size {2 ** self.m_max}.'
                              f' Note that error tolerances may no longer be satisfied',
                              MaxSamplesWarning)
                break

        self.data.time_integrate = time() - tstart
        # Approximate integral
        self.data.solution = muhat

        return muhat, self.data

    def _fwht_h(self, y):
        ytilde = np.squeeze(y)
        self.fwht.fwht_inplace(len(y), ytilde)
        return ytilde
        # ytilde = np.array(self.fwht_h_py(y), dtype=float)
        # return ytilde

    @staticmethod
    def _merge_fwht(ftilde_new, ftilde_next_new, mnext):
        ftilde_new = np.vstack([(ftilde_new + ftilde_next_new), (ftilde_new - ftilde_next_new)])
        return ftilde_new

    '''
    Digitally shift invariant kernel
    C1 : first row of the covariance matrix
    Lambda : eigen values of the covariance matrix
    Lambda_ring = fwht(C1 - 1)
    '''
    def kernel(self, xun, order, a, avoid_cancel_error, kern_type, debug_enable):
        kernel_func = CubBayesNetG.BuildKernelFunc(order)
        const_mult = 1/10

        if avoid_cancel_error:
            # Computes C1m1 = C1 - 1
            # C1_new = 1 + C1m1 indirectly computed in the process
            (vec_C1m1, C1_alt) = self.data.kernel_t(a * const_mult, kernel_func(xun))
            lambda_factor = max(abs(vec_C1m1))
            C1_alt = C1_alt / lambda_factor
            vec_C1m1 = vec_C1m1 / lambda_factor

            # eigenvalues must be real : Symmetric pos definite Kernel
            vec_lambda_ring = np.real(self._fwht_h(vec_C1m1.copy()))

            vec_lambda = vec_lambda_ring.copy()
            vec_lambda[0] = vec_lambda_ring[0] + len(vec_lambda_ring)/lambda_factor

            if debug_enable:
                # eigenvalues must be real : Symmetric pos definite Kernel
                vec_lambda_direct = np.real(np.array(self._fwht_h(C1_alt), dtype=float))  # Note: fwht output not normalized
                if sum(abs(vec_lambda_direct - vec_lambda)) > 1:
                    print('Possible error: check vec_lambda_ring computation')
        else:
            # direct approach to compute first row of the kernel Gram matrix
            vec_C1 = np.prod(1 + a * const_mult * kernel_func(xun), 2)
            # eigenvalues must be real : Symmetric pos definite Kernel
            vec_lambda = np.real(self._fwht_h(vec_C1))
            vec_lambda_ring = 0
            lambda_factor = 1

        return vec_lambda, vec_lambda_ring, lambda_factor

    # Builds High order walsh kernel function
    @staticmethod
    def BuildKernelFunc(order):
        # a1 = @(x)(-floor(log2(x)))
        def a1(x):
            out = -np.floor(log2(x + np.finfo(float).eps))
            out[x == 0] = 0  # a1 is zero when x is zero
            return out

        # t1 = @(x)(2.^(-a1(x)))
        def t1(x):
            out = 2**(-a1(x))
            out[x == 0] = 0  # t1 is zero when x is zero
            return out

        # t2 = @(x)(2.^(-2*a1(x)))
        def t2(x):
            out = (2**(-2 * a1(x)))
            out[x == 0] = 0  # t2 is zero when x is zero
            return out

        s1 = lambda x: (1 - 2 * x)
        s2 =lambda x: (1 / 3 - 2 * (1 - x) * x)
        ts2 = lambda x: ((1 - 5 * t1(x)) / 2 - (a1(x) - 2) * x)
        ts3 = lambda x: ((1 - 43 * t2(x)) / 18 + (5 * t1(x) - 1) * x + (a1(x) - 2) * x**2)

        if order == 1:
            kernFunc = lambda x: (6 * ((1 / 6) - 2**(np.floor(log2(x + np.finfo(float).eps)) - 1)))
        elif order == 2:
            omega2_1D =lambda x: (s1(x) + ts2(x))
            kernFunc = omega2_1D
        elif order == 3:
            omega3_1D = lambda x: (s1(x) + s2(x) + ts3(x))
            kernFunc = omega3_1D
        else:
            NotYetImplemented('cubBayesNet_g: kernel order not yet supported')

        return kernFunc

    '''
    @staticmethod
    def fwht_h_py(ynext):
        mnext = int(np.log2(len(ynext)))
        for l in range(mnext):
            nl = 2**l
            nmminlm1 = 2.**(mnext-l-1)
            ptind_nl = np.hstack(( np.tile(True,nl), np.tile(False,nl) ))
            ptind = np.tile(ptind_nl,int(nmminlm1))
            evenval = ynext[ptind]
            oddval = ynext[~ptind]
            ynext[ptind] = (evenval + oddval)
            ynext[~ptind] = (evenval - oddval)
        return ynext
    '''


class FWHT():
    def __init__(self):
        self.fwht_copy_cf = c_lib.fwht_copy
        self.fwht_copy_cf.argtypes = [
            ctypes.c_uint32,
            ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
            ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS')
        ]
        self.fwht_copy_cf.restype = None

        self.fwht_inplace_cf = c_lib.fwht_inplace
        self.fwht_inplace_cf.argtypes = [
            ctypes.c_uint32,
            ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
        ]
        self.fwht_inplace_cf.restype = None

    def fwht_copy(self, n, src, dst):
        self.fwht_copy_cf(n, src, dst)

    def fwht_inplace(self, n, src):
        self.fwht_inplace_cf(n, src)
