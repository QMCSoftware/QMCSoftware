from ._cub_bayes_ld_g import _CubBayesLDG
#from ..accumulate_data.ld_transform_bayes_data import LDTransformBayesData
from ..discrete_distribution import Lattice
from ..integrand import Keister
from ..util import ParameterError#, ParameterWarning #MaxSamplesWarning,
from numpy import log2
#from math import factorial
import numpy as np
#from time import time
#import warnings


class CubBayesLatticeG(_CubBayesLDG):
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
        indv_error      3.20e-04
        ci_low          1.808
        ci_high         1.809
        ci_comb_low     1.808
        ci_comb_high    1.809
        flags_comb      1
        flags_indv      1
        n_total         2^(8)
        n               2^(8)
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
	`GAIL cubBayesLattice_g <https://github.com/GailGithub/GAIL_Dev/blob/master/Algorithms/IntegrationExpectation/cubBayesLattice_g.m>`_.

    Reference
        [1] Sou-Cheng T. Choi, Yuhan Ding, Fred J. Hickernell, Lan Jiang, Lluis Antoni Jimenez Rugama,
        Da Li, Jagadeeswaran Rathinavel, Xin Tong, Kan Zhang, Yizhi Zhang, and Xuan Zhou,
        GAIL: Guaranteed Automatic Integration Library (Version 2.3) [MATLAB Software], 2019.
	Available from `GAIL <http://gailgithub.github.io/GAIL_Dev/>`_.

    Guarantee
        This algorithm attempts to calculate the integral of function f over the
        hyperbox [0,1]^d to a prescribed error tolerance tolfun:= max(abstol,
        reltol*| I |)
        with guaranteed confidence level, e.g., 99% when alpha=0.5%. If the
        algorithm terminates without showing any warning messages and provides
        an answer Q, then the following inequality would be satisfied:

                Pr(| Q - I | <= tolfun) = 99%.

        This Bayesian cubature algorithm guarantees for integrands that are considered
        to be an instance of a gaussian process that fall in the middle of samples space spanned.
        Where The sample space is spanned by the covariance kernel parametrized by the scale
        and shape parameter inferred from the sampled values of the integrand.
        For more details on how the covariance kernels are defined and the parameters are obtained,
        please refer to the references below.
    """

    def __init__(self, integrand, abs_tol=1e-2, rel_tol=0,
                 n_init=2 ** 8, n_max=2 ** 22, order=2, alpha=0.01, ptransform='C1sin',
                 error_fun=lambda sv, abs_tol, rel_tol: np.maximum(abs_tol, abs(sv) * rel_tol)):

        super(CubBayesLatticeG, self).__init__(integrand, fbt=self._fft, merge_fbt=self._merge_fft,
                                               ptransform=ptransform,
                                               allowed_distribs=[Lattice],
                                               kernel=self._shift_inv_kernel,
                                               abs_tol=abs_tol, rel_tol=rel_tol,
                                               n_init=n_init, n_max=n_max, alpha=alpha, error_fun=error_fun)

        self.parameters = ['abs_tol', 'rel_tol', 'n_init', 'n_max', 'order']
        # Set Attributes
        self.order = order  # Bernoulli kernel's order. If zero, choose order automatically

        self.use_gradient = False  # If true uses gradient descent in parameter search
        self.one_theta = True  # If true use common shape parameter for all dimensions
        # else allow shape parameter vary across dimensions
        self.ptransform = ptransform  # periodization transform
        self.errbd_type = 'MLE'  # Available options {'MLE', 'GCV', 'full'}

        # private properties
        # full_Bayes - Full Bayes - assumes m and s^2 as hyperparameters
        # GCV - Generalized cross validation
        self.kernType = 1  # Type-1: Bernoulli polynomial based algebraic convergence, Type-2: Truncated series

        if self.discrete_distrib.randomize == False:
            raise ParameterError("CubBayesLattice_g requires discrete_distrib to have randomize=True")
        if self.discrete_distrib.order != 'linear':
            raise ParameterError("CubBayesLattice_g requires discrete_distrib to have order='linear'")


    @staticmethod
    def _fft(y):
        ytilde = np.fft.fft(y)
        return ytilde

    @staticmethod
    def _fft_py(ynext):
        """
        Fast Fourier Transform (FFT) ynext, combine with y, then FFT all points.

        Args:
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

    def _shift_inv_kernel(self, xun, order, theta, avoid_cancel_error, kern_type, debug_enable):
        if kern_type == 1:
            b_order = order * 2  # Bernoulli polynomial order as per the equation
            #const_mult = -(-1) ** (b_order / 2) * ((2 * np.pi) ** b_order) / factorial(b_order)
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
            vec_lambda[0] = vec_lambda_ring[0] + len(vec_lambda_ring) / lambda_factor

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

class CubQMCBayesLatticeG(CubBayesLatticeG): pass
