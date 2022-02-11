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
    >>> data
    LDTransformBayesData (AccumulateData Object)
        solution        1.808
        error_bound     6.41e-04
        n_total         256
        time_integrate  ...
    CubBayesLatticeG (StoppingCriterion Object)
        abs_tol         0.050
        rel_tol         0
        n_init          2^(8)
        n_max           2^(22)
        order           2^(1)
    Keister (Integrand Object)
    Gaussian (TrueMeasure Object)
        mean            0
        covariance      2^(-1)
        decomp_type     PCA
    Lattice (DiscreteDistribution Object)
        d               2^(1)
        dvec            [0 1]
        randomize       1
        order           linear
        entropy         123456789
        spawn_key       ()
    
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
                 n_init=2 ** 8, n_max=2 ** 22, order=2, alpha=0.01, ptransform='C1sin',
                 error_fun=lambda sv,abs_tol,rel_tol: np.maximum(abs_tol,abs(sv)*rel_tol)):
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
        self.one_theta = True  # If true use common shape parameter for all dimensions
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

        # Sobol indices
        self.dprime = self.integrand.dprime
        self.cv = []
        self.ncv = len(self.cv)
        self.cast_complex = False
        self.d = self.discrete_distrib.d
        self.error_fun = error_fun

        # Verify Compliant Construction
        allowed_levels = ['single']
        allowed_distribs = [Lattice]
        allow_vectorized_integrals = True
        super(CubBayesLatticeG, self).__init__(allowed_levels, allowed_distribs, allow_vectorized_integrals)

        if self.discrete_distrib.randomize == False:
            raise ParameterError("CubBayesLattice_g requires discrete_distrib to have randomize=True")
        if self.discrete_distrib.order != 'linear':
            raise ParameterError("CubBayesLattice_g requires discrete_distrib to have order='linear'")

    def integrate_nd(self):
        t_start = time()
        self.datum = np.empty(self.dprime, dtype=object)
        for j in np.ndindex(self.dprime):
            self.datum[j] = LDTransformBayesData(self, self.integrand, self.true_measure, self.discrete_distrib,
                                             self.m_min, self.m_max, self._fft, self._merge_fft, self.kernel)

        self.data = LDTransformBayesData.__new__(LDTransformBayesData)
        self.data.flags_indv = np.tile(True, self.dprime)
        self.data.m = np.tile(self.m_min, self.dprime)
        self.data.n_min = 0
        self.data.ci_low = np.tile(-np.inf, self.dprime)
        self.data.ci_high = np.tile(np.inf, self.dprime)
        self.data.solution_indv = np.tile(np.nan, self.dprime)
        self.data.solution = np.nan
        self.data.xfull = np.empty((0, self.d))
        self.data.yfull = np.empty((0,) + self.dprime)
        stop_flag = np.tile(None, self.dprime)
        while True:
            m = self.data.m.max()
            n_min = self.data.n_min
            n_max = int(2 ** m)
            n = int(n_max - n_min)
            xnext, xnext_un = self.discrete_distrib.gen_samples(n_min=n_min, n_max=n_max, return_unrandomized=True, warn=False)
            ycvnext = np.empty((1 + self.ncv, n,) + self.dprime, dtype=float)
            ycvnext[0] = self.integrand.f(xnext, periodization_transform=self.ptransform,
                                          compute_flags=self.data.flags_indv)
            for k in range(self.ncv):
                ycvnext[1 + k] = self.cv[k].f(xnext, periodization_transform=self.ptransform,
                                              compute_flags=self.data.flags_indv)
            for j in np.ndindex(self.dprime):
                if not self.data.flags_indv[j]:
                    continue
                slice_yj = (0, slice(None),) + j
                y_val = ycvnext[slice_yj]

                # Update function values
                xnext_un_, ftilde_, m_ = self.datum[j].update_data(y_val_new=y_val, xnew=xnext, xunnew=xnext_un)
                success, muhat, r_order, err_bd = self.datum[j].stopping_criterion(xnext_un_, ftilde_, m_)
                bounds = muhat + np.array([-1, 1]) * err_bd
                stop_flag[j], self.data.solution_indv[j], self.data.ci_low[j], self.data.ci_high[j] = \
                    success, muhat, bounds[0], bounds[1]

            self.data.xfull = np.vstack((self.data.xfull, xnext))
            self.data.yfull = np.vstack((self.data.yfull, ycvnext[0]))
            self.data.indv_error = (self.data.ci_high - self.data.ci_low) / 2
            self.data.ci_comb_low, self.data.ci_comb_high, self.data.violated = self.integrand.bound_fun(
                self.data.ci_low, self.data.ci_high)
            error_low = self.error_fun(self.data.ci_comb_low, self.abs_tol, self.rel_tol)
            error_high = self.error_fun(self.data.ci_comb_high, self.abs_tol, self.rel_tol)
            self.data.solution = 1 / 2 * (self.data.ci_comb_low + self.data.ci_comb_high + error_low - error_high)
            rem_error_low = abs(self.data.ci_comb_low - self.data.solution) - error_low
            rem_error_high = abs(self.data.ci_comb_high - self.data.solution) - error_high
            self.data.flags_comb = np.maximum(rem_error_low, rem_error_high) >= 0
            self.data.flags_comb |= self.data.violated
            self.data.flags_indv = self.integrand.dependency(self.data.flags_comb)
            self.data.n = 2 ** self.data.m
            self.data.n_total = self.data.n.max()

            if np.sum(self.data.flags_indv)==0:
                break # stopping criterion met
            elif 2*self.data.n_total>self.n_max:
                # doubling samples would go over n_max
                warning_s = """
                Already generated %d samples.
                Trying to generate %d new samples would exceed n_max = %d.
                No more samples will be generated.
                Note that error tolerances may no longer be satisfied.""" \
                % (int(self.data.n_total),int(self.data.n_total),int(self.n_max))
                warnings.warn(warning_s, MaxSamplesWarning)
                break
            else:
                self.data.n_min = n_max
                self.data.m += self.data.flags_indv

        self.data.integrand = self.integrand
        self.data.true_measure = self.true_measure
        self.data.discrete_distrib = self.discrete_distrib
        self.data.stopping_crit = self
        self.data.parameters = [
            'solution',
            'indv_error',
            'ci_low',
            'ci_high',
            'ci_comb_low',
            'ci_comb_high',
            'flags_comb',
            'flags_indv',
            'n_total',
            'n',
            'time_integrate']
        self.data.datum = self.datum
        self.data.time_integrate = time()-t_start
        return self.data.solution,self.data

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
                warnings.warn('''
                    Already used maximum allowed sample size %d.
                    Note that error tolerances may no longer be satisfied.'''%(2**self.m_max),
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
