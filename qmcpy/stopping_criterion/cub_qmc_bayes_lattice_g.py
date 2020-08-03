from ._stopping_criterion import StoppingCriterion
from ..accumulate_data.ld_transform_bayes_data import LDTransformBayesData, OutParams
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
    
    >>> k = Keister(Gaussian(Lattice(2,randomize=False, linear=True),covariance=1./2))
    >>> sc = CubBayesLatticeG(k,abs_tol=.05)
    >>> solution,data = sc.integrate()
    >>> solution
    1.808134131740979
    >>> data
    Solution: 1.8081         
    Keister (Integrand Object)
    Lattice (DiscreteDistribution Object)
        dimension       2^(1)
        randomize       1
        seed            7
        backend         gail
        mimics          StdUniform
    Gaussian (TrueMeasure Object)
        mean            0
        covariance      2^(-1)
    CubBayesLatticeG (StoppingCriterion Object)
        abs_tol         0.050
        rel_tol         0
        n_init          2^(10)
        n_max           2^(35)
    LDTransformData (AccumulateData Object)
        n_total         2^(10)
        solution        1.808
        r_lag           2^(2)
        time_integrate  ...

    Adapted from 
        https://github.com/GailGithub/GAIL_Dev/blob/master/Algorithms/IntegrationExpectation/CubBayesLatticeG.m

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

        This Bayesian cubature algorithm guarantees for integrands that are considered to be an instance of a
        gaussian process that fall in the middle of samples space spanned.
        Where The sample space is spanned by the covariance kernel parametrized by the scale and shape parameter
        inferred from the sampled values of the integrand.
        For more details on how the covariance kernels are defined and the parameters are obtained, please
        refer to the references below.
    """
    parameters = ['abs_tol','rel_tol','n_init','n_max']

    def __init__(self, integrand, abs_tol=1e-2, rel_tol=0,
                 n_init=2 ** 8, n_max=2 ** 22, alpha=0.01, ptransform='C1sin'):
        # Set Attributes
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        m_min = log2(n_init)
        m_max = log2(n_max)
        if m_min % 1 != 0. or m_min < 8 or m_max % 1 != 0:
            warning_s = '''
                n_init and n_max must be a powers of 2.
                n_init must be >= 2^8.
                Using n_init = 2^10 and n_max=2^23.'''
            warnings.warn(warning_s, ParameterWarning)
            m_min = 10.
            m_max = 23.
        self.m_min = m_min
        self.m_max = m_max
        self.n_init = n_init  # number of samples to start with = 2^mmin
        self.n_max = n_max  # max number of samples allowed = 2^mmax
        self.alpha = alpha  # p-value, default 0.1%.
        self.order = 2  # Bernoulli kernel's order. If zero, choose order automatically

        # self.f = lambda x: x ** 2  # function to integrate
        self.integrand = integrand
        self.dim = integrand.dimension  # dimension of the integrand
        self.useGradient = False  # If true uses gradient descent in parameter search
        self.oneTheta = True  # If true use common shape parameter for all dimensions
        # else allow shape parameter vary across dimensions
        self.ptransform = ptransform  # periodization transform
        self.stopAtTol = True  # automatic mode: stop after meeting the error tolerance
        self.arbMean = True  # by default use zero mean algorithm
        self.stopCriterion = 'MLE'  # Available options {'MLE', 'GCV', 'full'}

        # private properties
        self.fullBayes = False  # Full Bayes - assumes m and s^2 as hyperparameters
        self.GCV = False  # Generalized cross validation
        self.vdc_order = False  # use Lattice points generated in vdc order
        self.kernType = 1  # Type-1: Bernoulli polynomial based algebraic convergence, Type-2: Truncated series

        self.avoidCancelError = True  # avoid cancellation error in stopping criterion
        self.uncert = 0  # quantile value for the error bound
        self.debugEnable = True  # enable debug prints

        # Credible interval : two-sided confidence, i.e., 1-alpha percent quantile
        if self.fullBayes:
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

        if distribution.randomize:
            raise ParameterError("CubBayesLattice_g requires distribution to have randomize=False")
        if distribution.backend != 'gail':
            raise ParameterError("CubBayesLattice_g requires distribution to have 'GAIL' backend")

    # computes the integral
    def integrate(self):
        # pick a random value to apply as shift
        shift = np.random.rand(1, self.dim)
        # shift = np.array([[0.58, 0.632]])

        # Construct AccumulateData Object to House Integration data
        self.data = LDTransformBayesData(self, self.integrand, self.m_min, self.m_max, shift, self.vdc_order)
        tstart = time()  # start the timer

        # Iteratively find the number of points required for the cubature to meet
        # the error threshold
        while True:
            tstart_iter = time()

            # Update function values
            xun_, ftilde_, m = self.data.update_data()
            stop_flag, muhat, order_, ErrBd = self.stopping_criterion(xun_, ftilde_, m)

            # if stopAtTol true, exit the loop
            # else, run for for all 'n' values.
            # Used to compute error values for 'n' vs error plotting
            if self.stopAtTol and stop_flag:
                break

            if m >= self.m_max:
                warnings.warn(f'Already used maximum allowed sample size {2 ** self.m_max}.'
                              f' Note that error tolerances may no longer be satisfied',
                              MaxSamplesWarning)
                break

        out = OutParams()
        out.n = 2 ** m
        out.time = time() - tstart
        out.ErrBd = ErrBd
        self.data.time_integrate = out.time
        ## Approximate integral
        self.data.solution = muhat

        if stop_flag == True:
            out.exitflag = 1
        else:
            out.exitflag = 2  # error tolerance may not be met

        return muhat, self.data

    # decides if the user-defined error threshold is met
    def stopping_criterion(self, xpts, ftilde, m):
        ftilde = ftilde.squeeze()
        n = 2 ** m
        success = False
        lnaRange = [-5, 5]
        r = self.order

        # search for optimal shape parameter
        lnaMLE = fminbnd(lambda lna: self.ObjectiveFunction(exp(lna), xpts, ftilde)[0],
                         x1=lnaRange[0], x2=lnaRange[1], xtol=1e-2, disp=0)

        aMLE = exp(lnaMLE)
        [loss, Lambda, Lambda_ring, RKHSnorm] = self.ObjectiveFunction(aMLE, xpts, ftilde)

        # Check error criterion
        # compute DSC
        if self.fullBayes == True:
            # full Bayes
            if self.avoidCancelError:
                DSC = abs(Lambda_ring[0] / n)
            else:
                DSC = abs((Lambda[0] / n) - 1)

            # 1-alpha two sided confidence interval
            ErrBd = self.uncert * sqrt(DSC * RKHSnorm / (n - 1))
        elif self.GCV == True:
            # GCV based stopping criterion
            if self.avoidCancelError:
                DSC = abs(Lambda_ring[0] / (n + Lambda_ring[0]))
            else:
                DSC = abs(1 - (n / Lambda[0]))

            temp = Lambda
            temp[0] = n + Lambda_ring[0]
            C_Inv_trace = sum(1. / temp(temp != 0))
            ErrBd = self.uncert * sqrt(DSC * (RKHSnorm) / C_Inv_trace)

        else:
            # empirical Bayes
            if self.avoidCancelError:
                DSC = abs(Lambda_ring[0] / (n + Lambda_ring[0]))
            else:
                DSC = abs(1 - (n / Lambda[0]))

            ErrBd = self.uncert * sqrt(DSC * RKHSnorm / n)

        if self.arbMean == True:  # zero mean case
            muhat = ftilde[0] / n
        else:  # non zero mean case
            muhat = ftilde[0] / Lambda[0]

        self.data.error_hat = ErrBd

        muhat = np.abs(muhat)
        muminus = muhat - ErrBd
        muplus = muhat + ErrBd

        if 2 * ErrBd <= max(self.abs_tol, self.rel_tol * abs(muminus)) + max(self.abs_tol, self.rel_tol * abs(muplus)):
            if ErrBd == 0:
                ErrBd = np.finfo(float).eps

            # stopping criterion achieved
            success = True

        return success, muhat, r, ErrBd

    # objective function to estimate parameter theta
    # MLE : Maximum likelihood estimation
    # GCV : Generalized cross validation
    def ObjectiveFunction(self, a, xun, ftilde):

        n = len(ftilde)
        [Lambda, Lambda_ring] = self.kernel(xun, self.order, a, self.avoidCancelError,
                                            self.kernType, self.debugEnable)

        Lambda = abs(Lambda)
        # compute RKHSnorm
        temp = abs(ftilde[Lambda != 0] ** 2) / (Lambda[Lambda != 0])

        # compute loss
        if self.GCV == True:
            # GCV
            temp_gcv = abs(ftilde[Lambda != 0] / (Lambda[Lambda != 0])) ** 2
            loss1 = 2 * log(sum(1. / Lambda[Lambda != 0]))
            loss2 = log(sum(temp_gcv[1:]))
            # ignore all zero eigenvalues
            loss = loss2 - loss1

            if self.arbMean == True:
                RKHSnorm = sum(temp_gcv[1:]) / n
            else:
                RKHSnorm = sum(temp_gcv) / n
        else:
            # default: MLE
            if self.arbMean == True:
                RKHSnorm = sum(temp[1:]) / n
                temp_1 = sum(temp[1:])
            else:
                RKHSnorm = sum(temp) / n
                temp_1 = sum(temp)

            # ignore all zero eigenvalues
            loss1 = sum(log(abs(Lambda[Lambda != 0])))
            loss2 = n * log(temp_1)
            loss = loss1 + loss2

        if self.debugEnable:
            self.data.alertMsg(loss1, 'Inf', 'Imag')
            self.data.alertMsg(RKHSnorm, 'Imag')
            self.data.alertMsg(loss2, 'Inf', 'Imag')
            self.data.alertMsg(loss, 'Inf', 'Imag', 'Nan')
            self.data.alertMsg(Lambda, 'Imag')

        return loss, Lambda, Lambda_ring, RKHSnorm

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
    def kernel(xun, order, a, avoidCancelError, kernType, debug_enable):

        if kernType == 1:
            b_order = order * 2  # Bernoulli polynomial order as per the equation
            constMult = -(-1) ** (b_order / 2) * ((2 * np.pi) ** b_order) / factorial(b_order)
            # constMult = -(-1)**(b_order/2)
            if b_order == 2:
                bernPoly = lambda x: (-x * (1 - x) + 1 / 6)
            elif b_order == 4:
                bernPoly = lambda x: (((x * (1 - x)) ** 2) - 1 / 30)
            else:
                print('Error: Bernoulli order not implemented !')

            kernelFunc = lambda x: bernPoly(x)
        else:
            b = order
            kernelFunc = lambda x: 2 * b * ((np.cos(2 * np.pi * x) - b)) / (1 + b ** 2 - 2 * b * np.cos(2 * np.pi * x))
            constMult = 1

        if avoidCancelError:
            # Computes C1m1 = C1 - 1
            # C1_new = 1 + C1m1 indirectly computed in the process
            (C1m1, C1_alt) = CubBayesLatticeG.kernel_t(a * constMult, kernelFunc(xun))
            # eigenvalues must be real : Symmetric pos definite Kernel
            Lambda_ring = np.real(np.fft.fft(C1m1))

            Lambda = Lambda_ring.copy()
            Lambda[0] = Lambda_ring[0] + len(Lambda_ring)
            # C1 = prod(1 + (a)*constMult*bernPoly(xun),2)  # direct computation
            # Lambda = real(fft(C1))

            if debug_enable == True:
                # eigenvalues must be real : Symmetric pos definite Kernel
                Lambda_direct = np.real(np.fft.fft(C1_alt))  # Note: fft output unnormalized
                if sum(abs(Lambda_direct - Lambda)) > 1:
                    print('Possible error: check Lambda_ring computation')

        else:
            # direct approach to compute first row of the kernel Gram matrix
            C1 = np.prod(1 + (a) * constMult * kernelFunc(xun), 2)
            # matlab's builtin fft is much faster and accurate
            # eigenvalues must be real : Symmetric pos definite Kernel
            Lambda = np.real(np.fft.fft(C1))
            Lambda_ring = 0

        return Lambda, Lambda_ring
