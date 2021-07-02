from ._stopping_criterion import StoppingCriterion
from ..accumulate_data.ld_transform_bayes_data import LDTransformBayesData
from ..discrete_distribution import Lattice
from ..integrand import Keister
from ..util import MaxSamplesWarning, ParameterError, ParameterWarning
from numpy import log2
from math import factorial
import numpy as np
from time import time
import warnings


class CubBayesLatticeG(StoppingCriterion):
    """
    Stopping criterion for Bayesian Cubature using rank-1 Lattice sequence with guaranteed
    accuracy over a d-dimensional region to integrate within a specified generalized error
    tolerance with guarantees under Bayesian assumptions.
    
    >>> k = Keister(Lattice(2, order='linear', seed=123456789))
    >>> sc = CubBayesLatticeG(k,abs_tol=.05)
    >>> solution,data = sc.integrate()
    >>> solution
    1.808...
    >>> data
    Solution: 1.8082         
    Keister (Integrand Object)
    Lattice (DiscreteDistribution Object)
        d               2^(1)
        randomize       1
        order           linear
        seed            123456789
        mimics          StdUniform
    Gaussian (TrueMeasure Object)
        mean            0
        covariance      2^(-1)
        decomp_type     pca
    CubBayesLatticeG (StoppingCriterion Object)
        abs_tol         0.050
        rel_tol         0
        n_init          2^(8)
        n_max           2^(22)
        order           2^(1)
    LDTransformBayesData (AccumulateData Object)
        n_total         256
        solution        1.808
        error_bound     7.37e-04
        time_integrate  ...
    
    Adapted from 
        https://github.com/GailGithub/GAIL_Dev/blob/master/Algorithms/IntegrationExpectation/cubBayesLattice_g.m

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
                 n_init=2 ** 8, n_max=2 ** 22, order=2, alpha=0.01, ptransform='C1sin'):
        self.parameters = ['abs_tol', 'rel_tol', 'n_init', 'n_max', 'order']
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
            m_min = 8
            m_max = 22
        self.m_min = m_min
        self.m_max = m_max
        self.n_init = n_init  # number of samples to start with = 2^mmin
        self.n_max = n_max  # max number of samples allowed = 2^mmax
        self.alpha = alpha  # p-value, default 0.1%.
        self.order = order  # Bernoulli kernel's order. If zero, choose order automatically

        self.use_gradient = False  # If true uses gradient descent in parameter search
        self.one_theta = False  # If true use common shape parameter for all dimensions
        # else allow shape parameter vary across dimensions
        self.ptransform = ptransform  # periodization transform
        self.stop_at_tol = True  # automatic mode: stop after meeting the error tolerance
        self.arb_mean = True  # by default use zero mean algorithm
        self.errbd_type = 'MLE'  # Available options {'MLE', 'GCV', 'full'}

        # private properties
        # full_Bayes - Full Bayes - assumes m and s^2 as hyperparameters
        # GCV - Generalized cross validation
        self.kernType = 1  # Type-1: Bernoulli polynomial based algebraic convergence, Type-2: Truncated series

        self.avoid_cancel_error = True  # avoid cancellation error in stopping criterion
        self.debug_enable = False  # enable debug prints
        self.data = None

        # QMCPy Objs
        self.integrand = integrand
        self.true_measure = self.integrand.true_measure
        self.discrete_distrib = self.integrand.discrete_distrib
        
        # Verify Compliant Construction
        allowed_levels = ['single']
        allowed_distribs = ["Lattice"]
        allow_vectorized_integrals = False
        super(CubBayesLatticeG, self).__init__(allowed_levels, allowed_distribs, allow_vectorized_integrals)

        if self.discrete_distrib.randomize == False:
            raise ParameterError("CubBayesLattice_g requires discrete_distrib to have randomize=True")
        if self.discrete_distrib.order != 'linear':
            raise ParameterError("CubBayesLattice_g requires discrete_distrib to have order='linear'")

    # computes the integral
    def integrate(self):
        # Construct AccumulateData Object to House Integration data
        self.data = LDTransformBayesData(self, self.integrand, self.true_measure, self.discrete_distrib,
            self.m_min, self.m_max, self._fft, self._merge_fft, self.kernel)
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

    @staticmethod
    def _fft(y):
        ytilde = np.fft.fft(y)
        return ytilde

    @staticmethod
    def _fft_py(ynext):
        """
        Fast Fourier Transform (FFT) ynext, combine with y, then FFT all points.

        Args:
            y (ndarray): all previous samples
            ynext (ndarray): next samples

        Return:
            ndarray: y and ynext combined and transformed
        """
        y = np.array([], dtype=complex)
        # y = y.astype(complex)
        ynext = ynext.astype(complex)
        ## Compute initial FFT on next points
        mnext = int(log2(len(ynext)))
        for l in range(mnext):
            nl = 2 ** l
            nmminlm1 = 2 ** (mnext - l - 1)
            ptind_nl = np.hstack((np.tile(True, nl), np.tile(False, nl)))
            ptind = np.tile(ptind_nl, int(nmminlm1))
            coef = np.exp(-2. * np.pi * 1j * np.arange(nl) / (2 * nl))
            coefv = np.tile(coef, int(nmminlm1))
            evenval = ynext[ptind]
            oddval = ynext[~ptind]
            ynext[ptind] = (evenval + coefv * oddval) / 2.
            ynext[~ptind] = (evenval - coefv * oddval) / 2.
        y = np.hstack((y, ynext))
        if len(y) > len(ynext):  # already generated some samples samples
            ## Compute FFT on all points
            nl = 2 ** mnext
            ptind = np.hstack((np.tile(True, int(nl)), np.tile(False, int(nl))))
            coefv = np.exp(-2 * np.pi * 1j * np.arange(nl) / (2 * nl))
            evenval = y[ptind]
            oddval = y[~ptind]
            y[ptind] = (evenval + coefv * oddval) / 2.
            y[~ptind] = (evenval - coefv * oddval) / 2.
        return y

    @staticmethod
    def _merge_fft(ftilde_new, ftilde_next_new, mnext):
        # using FFT butterfly plot technique merges two halves of fft
        ftilde_new = np.vstack([ftilde_new, ftilde_next_new])
        nl = 2 ** mnext
        ptind = np.ndarray(shape=(2 * nl, 1), buffer=np.array([True] * nl + [False] * nl), dtype=bool)
        coef = np.exp(-2 * np.pi * 1j * np.ndarray(shape=(nl, 1), buffer=np.arange(0, nl), dtype=int) / (2 * nl))
        coefv = np.tile(coef, (1, 1))
        evenval = ftilde_new[ptind].reshape((nl, 1))
        oddval = ftilde_new[~ptind].reshape((nl, 1))
        ftilde_new[ptind] = np.squeeze(evenval + coefv * oddval)
        ftilde_new[~ptind] = np.squeeze(evenval - coefv * oddval)
        return ftilde_new

    '''
    Shift invariant kernel
    C1 : first row of the covariance matrix
    Lambda : eigen values of the covariance matrix
    Lambda_ring = fft(C1 - 1)
    '''
    def kernel(self, xun, order, theta, avoid_cancel_error, kern_type, debug_enable):
        if kern_type == 1:
            b_order = order * 2  # Bernoulli polynomial order as per the equation
            const_mult = -(-1) ** (b_order / 2) * ((2 * np.pi) ** b_order) / factorial(b_order)
            const_mult = -(-1) ** (b_order / 2)
            if b_order == 2:
                bern_poly = lambda x: (-x * (1 - x) + 1 / 6)
            elif b_order == 4:
                bern_poly = lambda x: (((x * (1 - x)) ** 2) - 1 / 30)
            else:
                print('Error: Bernoulli order not implemented !')

            kernel_func = lambda x: bern_poly(x)
        else:
            b = order
            kernel_func = lambda x: 2 * b * (np.cos(2 * np.pi * x) - b) / (1 + b ** 2 - 2 * b * np.cos(2 * np.pi * x))
            const_mult = 1

        if avoid_cancel_error:
            # Computes C1m1 = C1 - 1
            # C1_new = 1 + C1m1 indirectly computed in the process
            (vec_C1m1, C1_alt) = self.data.kernel_t(theta * const_mult, kernel_func(xun))

            lambda_factor = max(abs(vec_C1m1))
            C1_alt = C1_alt / lambda_factor
            vec_C1m1 = vec_C1m1 / lambda_factor
            # eigenvalues must be real : Symmetric pos definite Kernel
            vec_lambda_ring = np.real(CubBayesLatticeG._fft(vec_C1m1))

            vec_lambda = vec_lambda_ring.copy()
            vec_lambda[0] = vec_lambda_ring[0] + len(vec_lambda_ring)/lambda_factor

            if debug_enable:
                # eigenvalues must be real : Symmetric pos definite Kernel
                vec_lambda_direct = np.real(CubBayesLatticeG._fft(C1_alt))  # Note: fft output unnormalized
                if sum(abs(vec_lambda_direct - vec_lambda)) > 1:
                    print('Possible error: check vec_lambda_ring computation')
        else:
            # direct approach to compute first row of the kernel Gram matrix
            vec_C1 = np.prod(1 + theta * const_mult * kernel_func(xun), 2)
            # matlab's builtin fft is much faster and accurate
            # eigenvalues must be real : Symmetric pos definite Kernel
            vec_lambda = np.real(CubBayesLatticeG._fft(vec_C1))
            vec_lambda_ring = 0
            lambda_factor = 1

        return vec_lambda, vec_lambda_ring, lambda_factor
