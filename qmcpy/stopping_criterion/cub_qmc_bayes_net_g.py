from ._stopping_criterion import StoppingCriterion
from ..accumulate_data.ld_transform_bayes_data import LDTransformBayesData
from ..discrete_distribution import Sobol
from ..true_measure import Gaussian
from ..integrand import Keister
from ..util import MaxSamplesWarning, ParameterError, ParameterWarning, NotYetImplemented
from ..discrete_distribution.c_lib import c_lib
import ctypes
from numpy import sqrt, log2, exp, log, ctypeslib
from math import factorial
import numpy as np
from time import time
from scipy.optimize import fminbound as fminbnd
from scipy.stats import norm as gaussnorm
from scipy.stats import t as tnorm
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
    1.811...
    >>> data
    Solution: 1.8110         
    Keister (Integrand Object)
    Sobol (DiscreteDistribution Object)
        d               2^(1)
        randomize       1
        graycode        0
        seed            [77469 77107]
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
        solution        1.811
        error_bound     0.015
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
                n_init must be >= 2^5.
                Using n_init = 2^10 and n_max=2^22.'''
            warnings.warn(warning_s, ParameterWarning)
            m_min = 5.
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
        self.stopCriterion = 'MLE'  # Available options {'MLE', 'GCV', 'full'}

        # private properties
        self.full_Bayes = False  # Full Bayes - assumes m and s^2 as hyperparameters
        self.GCV = False  # Generalized cross validation
        self.kernType = 1  # Type-1:

        self.avoid_cancel_error = True  # avoid cancellation error in stopping criterion
        self.uncert = 0  # quantile value for the error bound
        self.debug_enable = True  # enable debug prints
        self.data = None
        self.fwht = FWHT()

        # Credible interval : two-sided confidence, i.e., 1-alpha percent quantile
        if self.full_Bayes:
            # degrees of freedom = 2^mmin - 1
            self.uncert = -tnorm.ppf(self.alpha / 2, (2 ** self.m_min) - 1)
        else:
            self.uncert = -gaussnorm.ppf(self.alpha / 2)

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
            self.m_min, self.m_max, self._fwht_h, self._merge_fwht)
        tstart = time()  # start the timer

        # Iteratively find the number of points required for the cubature to meet
        # the error threshold
        while True:
            # Update function values
            xun_, ftilde_, m = self.data.update_data()
            stop_flag, muhat, order_, err_bnd = self.stopping_criterion(xun_, ftilde_, m)

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

    # decides if the user-defined error threshold is met
    def stopping_criterion(self, xpts, ftilde, m):
        ftilde = ftilde.squeeze()
        n = 2 ** m
        success = False
        lna_range = [-5, 5]
        r = self.order

        # search for optimal shape parameter
        lna_MLE = fminbnd(lambda lna: self.objective_function(exp(lna), xpts, ftilde)[0],
                          x1=lna_range[0], x2=lna_range[1], xtol=1e-2, disp=0)
        aMLE = exp(lna_MLE)
        _, vec_lambda, vec_lambda_ring, RKHS_norm = self.objective_function(aMLE, xpts, ftilde)

        # Check error criterion
        # compute DSC
        if self.full_Bayes:
            # full Bayes
            if self.avoid_cancel_error:
                DSC = abs(vec_lambda_ring[0] / n)
            else:
                DSC = abs((vec_lambda[0] / n) - 1)
            # 1-alpha two sided confidence interval
            err_bd = self.uncert * sqrt(DSC * RKHS_norm / (n - 1))
        elif self.GCV:
            # GCV based stopping criterion
            if self.avoid_cancel_error:
                DSC = abs(vec_lambda_ring[0] / (n + vec_lambda_ring[0]))
            else:
                DSC = abs(1 - (n / vec_lambda[0]))
            temp = vec_lambda
            temp[0] = n + vec_lambda_ring[0]
            mC_inv_trace = sum(1. / temp(temp != 0))
            err_bd = self.uncert * sqrt(DSC * RKHS_norm / mC_inv_trace)
        else:
            # empirical Bayes
            if self.avoid_cancel_error:
                DSC = abs(vec_lambda_ring[0] / (n + vec_lambda_ring[0]))
            else:
                DSC = abs(1 - (n / vec_lambda[0]))
            err_bd = self.uncert * sqrt(DSC * RKHS_norm / n)

        if self.arb_mean:  # zero mean case
            muhat = ftilde[0] / n
        else:  # non zero mean case
            muhat = ftilde[0] / vec_lambda[0]

        self.data.error_bound = err_bd
        muhat = np.abs(muhat)
        muminus = muhat - err_bd
        muplus = muhat + err_bd

        if 2 * err_bd <= max(self.abs_tol, self.rel_tol * abs(muminus)) + max(self.abs_tol, self.rel_tol * abs(muplus)):
            if err_bd == 0:
                err_bd = np.finfo(float).eps

            # stopping criterion achieved
            success = True

        return success, muhat, r, err_bd

    # objective function to estimate parameter theta
    # MLE : Maximum likelihood estimation
    # GCV : Generalized cross validation
    def objective_function(self, a, xun, ftilde):
        n = len(ftilde)
        [vec_lambda, vec_lambda_ring] = self.kernel(xun, self.order, a, self.avoid_cancel_error,
                                                    self.kernType, self.debug_enable)

        vec_lambda = abs(vec_lambda)
        # compute RKHS_norm
        temp = abs(ftilde[vec_lambda != 0] ** 2) / (vec_lambda[vec_lambda != 0])

        # compute loss
        if self.GCV:
            # GCV
            temp_gcv = abs(ftilde[vec_lambda != 0] / (vec_lambda[vec_lambda != 0])) ** 2
            loss1 = 2 * log(sum(1. / vec_lambda[vec_lambda != 0]))
            loss2 = log(sum(temp_gcv[1:]))
            # ignore all zero eigenvalues
            loss = loss2 - loss1

            if self.arb_mean:
                RKHS_norm = sum(temp_gcv[1:]) / n
            else:
                RKHS_norm = sum(temp_gcv) / n
        else:
            # default: MLE
            if self.arb_mean:
                RKHS_norm = sum(temp[1:]) / n
                temp_1 = sum(temp[1:])
            else:
                RKHS_norm = sum(temp) / n
                temp_1 = sum(temp)

            # ignore all zero eigenvalues
            loss1 = sum(log(abs(vec_lambda[vec_lambda != 0])))
            loss2 = n * log(temp_1)
            loss = loss1 + loss2

        if self.debug_enable:
            self.data.alert_msg(loss1, 'Inf', 'Imag')
            self.data.alert_msg(RKHS_norm, 'Imag')
            self.data.alert_msg(loss2, 'Inf', 'Imag')
            self.data.alert_msg(loss, 'Inf', 'Imag', 'Nan')
            self.data.alert_msg(vec_lambda, 'Imag')

        return loss, vec_lambda, vec_lambda_ring, RKHS_norm

    # Computes modified kernel Km1 = K - 1
    # Useful to avoid cancellation error in the computation of (1 - n/\lambda_1)
    @staticmethod
    def kernel_t(aconst, Bern):
        theta = aconst
        d = np.size(Bern, 1)

        Kjm1 = theta * Bern[:, 0]  # Kernel at j-dim minus One
        Kj = 1 + Kjm1  # Kernel at j-dim

        for j in range(1, d):
            Kjm1_prev = Kjm1
            Kj_prev = Kj  # save the Kernel at the prev dim

            Kjm1 = theta * Bern[:, j] * Kj_prev + Kjm1_prev
            Kj = 1 + Kjm1

        Km1 = Kjm1
        K = Kj
        return [Km1, K]

    '''
    Digitally shift invariant kernel
    C1 : first row of the covariance matrix
    Lambda : eigen values of the covariance matrix
    Lambda_ring = fwht(C1 - 1)
    '''
    def kernel(self, xun, order, a, avoid_cancel_error, kern_type, debug_enable):
        kernel_func = CubBayesNetG.BuildKernelFunc(order)
        const_mult = 1

        if avoid_cancel_error:
            # Computes C1m1 = C1 - 1
            # C1_new = 1 + C1m1 indirectly computed in the process
            (vec_C1m1, C1_alt) = CubBayesNetG.kernel_t(a * const_mult, kernel_func(xun))
            # eigenvalues must be real : Symmetric pos definite Kernel
            vec_lambda_ring = np.real(self._fwht_h(vec_C1m1.copy()))

            vec_lambda = vec_lambda_ring.copy()
            vec_lambda[0] = vec_lambda_ring[0] + len(vec_lambda_ring)

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

        return vec_lambda, vec_lambda_ring

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
