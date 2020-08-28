from ._stopping_criterion import StoppingCriterion
from ..accumulate_data.ld_transform_bayes_data import LDTransformBayesData
from ..discrete_distribution import Lattice
from ..true_measure import Gaussian
from ..integrand import Keister
from ..util import MaxSamplesWarning, ParameterError, ParameterWarning
from numpy import sqrt, log2, exp, log
from math import factorial
import numpy as np
from time import time
from scipy.optimize import fminbound as fminbnd
from scipy.stats import norm as gaussnorm
from scipy.stats import t as tnorm
import warnings


class CubBayesLatticeG(StoppingCriterion):
    """
    Stopping criterion for Bayesian Cubature using rank-1 Lattice sequence with guaranteed
    accuracy over a d-dimensional region to integrate within a specified generalized error
    tolerance with guarantees under Bayesian assumptions.
    
    >>> k = Keister(Gaussian(Lattice(2, linear=True, seed=123456789, backend='GAIL'),covariance=1./2))
    >>> sc = CubBayesLatticeG(k,abs_tol=.05)
    >>> solution,data = sc.integrate()
    >>> solution
    1.80818168796735...
    >>> data
    Solution: 1.8082...
    Keister (Integrand Object)
    Lattice (DiscreteDistribution Object)
        dimension       2^(1)
        randomize       1
        seed            123456789
        mimics          StdUniform
        backend         GAIL
    Gaussian (TrueMeasure Object)
        mean            0
        covariance      2^(-1)
        decomp_type     pca
    CubBayesLatticeG (StoppingCriterion Object)
        abs_tol         0.050
        rel_tol         0
        n_init          2^(8)
        n_max           2^(22)
    LDTransformBayesData (AccumulateData Object)
        n_total         256
        solution        1.808
        error_bound     7.36e-04
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
    parameters = ['abs_tol', 'rel_tol', 'n_init', 'n_max']

    def __init__(self, integrand, abs_tol=1e-2, rel_tol=0,
                 n_init=2 ** 8, n_max=2 ** 22, alpha=0.01, ptransform='C1sin'):
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
        self.order = 2  # Bernoulli kernel's order. If zero, choose order automatically

        self.integrand = integrand
        self.dim = integrand.dimension  # dimension of the integrand
        self.useGradient = False  # If true uses gradient descent in parameter search
        self.oneTheta = True  # If true use common shape parameter for all dimensions
        # else allow shape parameter vary across dimensions
        self.ptransform = ptransform  # periodization transform
        self.stop_at_tol = True  # automatic mode: stop after meeting the error tolerance
        self.arb_mean = True  # by default use zero mean algorithm
        self.stopCriterion = 'MLE'  # Available options {'MLE', 'GCV', 'full'}

        # private properties
        self.full_Bayes = False  # Full Bayes - assumes m and s^2 as hyperparameters
        self.GCV = False  # Generalized cross validation
        self.kernType = 1  # Type-1: Bernoulli polynomial based algebraic convergence, Type-2: Truncated series

        self.avoid_cancel_error = True  # avoid cancellation error in stopping criterion
        self.uncert = 0  # quantile value for the error bound
        self.debug_enable = True  # enable debug prints
        self.data = None

        # Credible interval : two-sided confidence, i.e., 1-alpha percent quantile
        if self.full_Bayes:
            # degrees of freedom = 2^mmin - 1
            self.uncert = -tnorm.ppf(self.alpha / 2, (2 ** self.m_min) - 1)
        else:
            self.uncert = -gaussnorm.ppf(self.alpha / 2)

        # Verify Compliant Construction
        distribution = integrand.measure.distribution
        self.distribution = distribution
        allowed_levels = ['single']
        allowed_distribs = ["Lattice"]
        super(CubBayesLatticeG, self).__init__(distribution, integrand, allowed_levels, allowed_distribs)

        if distribution.randomize == False:
            raise ParameterError("CubBayesLattice_g requires distribution to have randomize=True")
        if not distribution.linear:
            raise ParameterError("CubBayesLattice_g requires distribution to have linear=True")
        if distribution.backend != 'GAIL':
            raise ParameterError("CubBayesLattice_g requires distribution to have 'GAIL' backend")

    # computes the integral
    def integrate(self):
        # Construct AccumulateData Object to House Integration data
        self.data = LDTransformBayesData(self, self.integrand, self.m_min, self.m_max)
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
    Shift invariant kernel
    C1 : first row of the covariance matrix
    Lambda : eigen values of the covariance matrix
    Lambda_ring = fft(C1 - 1)
    '''

    @staticmethod
    def kernel(xun, order, a, avoid_cancel_error, kern_type, debug_enable):

        if kern_type == 1:
            b_order = order * 2  # Bernoulli polynomial order as per the equation
            const_mult = -(-1) ** (b_order / 2) * ((2 * np.pi) ** b_order) / factorial(b_order)
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
            (vec_C1m1, C1_alt) = CubBayesLatticeG.kernel_t(a * const_mult, kernel_func(xun))
            # eigenvalues must be real : Symmetric pos definite Kernel
            vec_lambda_ring = np.real(np.fft.fft(vec_C1m1))

            vec_lambda = vec_lambda_ring.copy()
            vec_lambda[0] = vec_lambda_ring[0] + len(vec_lambda_ring)

            if debug_enable:
                # eigenvalues must be real : Symmetric pos definite Kernel
                vec_lambda_direct = np.real(np.fft.fft(C1_alt))  # Note: fft output unnormalized
                if sum(abs(vec_lambda_direct - vec_lambda)) > 1:
                    print('Possible error: check vec_lambda_ring computation')
        else:
            # direct approach to compute first row of the kernel Gram matrix
            vec_C1 = np.prod(1 + a * const_mult * kernel_func(xun), 2)
            # matlab's builtin fft is much faster and accurate
            # eigenvalues must be real : Symmetric pos definite Kernel
            vec_lambda = np.real(np.fft.fft(vec_C1))
            vec_lambda_ring = 0

        return vec_lambda, vec_lambda_ring
