from ._cub_bayes_ld_g import _CubBayesLDG
from ..accumulate_data.ld_transform_bayes_data import LDTransformBayesData
from ..discrete_distribution import DigitalNetB2
from ..integrand import Keister
from ..util import MaxSamplesWarning, ParameterError, ParameterWarning, NotYetImplemented
from ..discrete_distribution._c_lib import _c_lib
import ctypes
import numpy as np
from time import time
import warnings


class CubBayesNetG(_CubBayesLDG):
    r"""
    Stopping criterion for Bayesian Cubature using digital net sequence with guaranteed
    accuracy over a d-dimensional region to integrate within a specified generalized error
    tolerance with guarantees under Bayesian assumptions.

    >>> k = Keister(DigitalNetB2(2, seed=123456789))
    >>> sc = CubBayesNetG(k,abs_tol=.05)
    >>> solution,data = sc.integrate()
    >>> data
    LDTransformBayesData (AccumulateData Object)
        solution        1.804
        comb_bound_low  1.786
        comb_bound_high 1.821
        comb_flags      1
        n_total         2^(8)
        n               2^(8)
        time_integrate  ...
    CubBayesNetG (StoppingCriterion Object)
        abs_tol         0.050
        rel_tol         0
        n_init          2^(8)
        n_max           2^(22)
    Keister (Integrand Object)
    Gaussian (AbstractTrueMeasure Object)
        mean            0
        covariance      2^(-1)
        decomp_type     PCA
    DigitalNetB2 (AbstractDiscreteDistribution Object)
        d               2^(1)
        dvec            [0 1]
        randomize       LMS_DS
        graycode        0
        entropy         123456789
        spawn_key       ()

    Adapted from `GAIL cubBayesNet_g <https://github.com/GailGithub/GAIL_Dev/blob/master/Algorithms/IntegrationExpectation/cubBayesNet_g.m>`_.

    Guarantee:
        This algorithm attempts to calculate the integral of function :math:`f` over the
        hyperbox :math:`[0,1]^d` to a prescribed error tolerance :math:`\mbox{tolfun} := max(\mbox{abstol},
        \mbox{reltol}*| I |)` with a guaranteed confidence level, e.g., :math:`99\%` when alpha= :math:`0.5\%`.
        If the algorithm terminates without showing any warning messages and provides
        an answer :math:`Q`, then the following inequality would be satisfied:

        .. math::
                Pr(| Q - I | <= \mbox{tolfun}) = 99\%.

        This Bayesian cubature algorithm guarantees for integrands that are considered
        to be an instance of a Gaussian process that falls in the middle of samples space spanned.
        Where The sample space is spanned by the covariance kernel parametrized by the scale
        and shape parameter inferred from the sampled values of the integrand.
        For more details on how the covariance kernels are defined and the parameters are obtained,
        please refer to the references below.

    References:
        [1] Jagadeeswaran Rathinavel, Fast automatic Bayesian cubature using matching kernels and designs,
        PhD thesis, Illinois Institute of Technology, 2019.

        [2] Sou-Cheng T. Choi, Yuhan Ding, Fred J. Hickernell, Lan Jiang, Lluis Antoni Jimenez Rugama,
        Da Li, Jagadeeswaran Rathinavel, Xin Tong, Kan Zhang, Yizhi Zhang, and Xuan Zhou,
        GAIL: Guaranteed Automatic Integration Library (Version 2.3) [MATLAB Software], 2019.
        Available from `GAIL <http://gailgithub.github.io/GAIL_Dev/>`_.

    """

    def __init__(self, integrand, abs_tol=1e-2, rel_tol=0,
                 n_init=2 ** 8, n_max=2 ** 22, alpha=0.01,
                 error_fun=lambda sv, abs_tol, rel_tol: np.maximum(abs_tol, abs(sv) * rel_tol)):
        """
        Args:
            integrand (Integrand): an instance of Integrand
            abs_tol (np.ndarray): absolute error tolerance
            rel_tol (np.ndarray): relative error tolerance
            n_init (int): initial number of samples
            n_max (int): maximum number of samples
            alpha (float): significance level or p-value
            error_fun: function taking in the approximate solution vector,
                absolute tolerance, and relative tolerance which returns the approximate error.
                Default indicates integration until either absolute OR relative tolerance is satisfied.
        """
        super(CubBayesNetG, self).__init__(integrand, fbt=self._fwht_h, merge_fbt=self._merge_fwht,
                                           ptransform=None,
                                           allowed_distribs=[DigitalNetB2],
                                           kernel=self._shift_inv_kernel_digital,
                                           abs_tol=abs_tol, rel_tol=rel_tol,
                                           n_init=n_init, n_max=n_max, alpha=alpha, error_fun=error_fun)

        self.parameters = ['abs_tol', 'rel_tol', 'n_init', 'n_max']
        # Set Attributes
        self.order = 1  # Currently supports only order=1

        self.use_gradient = False  # If true uses gradient descent in parameter search
        self.one_theta = True  # If true use common shape parameter for all dimensions
        # else allow shape parameter vary across dimensions
        self.errbd_type = 'MLE'  # Available options {'MLE', 'GCV', 'full'}

        # private properties
        # Full Bayes - assumes m and s^2 as hyperparameters
        # GCV - Generalized cross validation
        self.kernType = 1  # Type-1:

        self.uncert = 0  # quantile value for the error bound
        self.data = None
        self.fwht = FWHT()

        if self.discrete_distrib.randomize == "FALSE":
            raise ParameterError("CubBayesNet_g requires discrete_distrib to have randomize=True")


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

    def _shift_inv_kernel_digital(self, xun, order, a, avoid_cancel_error, kern_type, debug_enable):
        kernel_func = CubBayesNetG.BuildKernelFunc(order)
        const_mult = 1 / 10

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
            vec_lambda[0] = vec_lambda_ring[0] + len(vec_lambda_ring) / lambda_factor

            if debug_enable:
                # eigenvalues must be real : Symmetric pos definite Kernel
                vec_lambda_direct = np.real(
                    np.array(self._fwht_h(C1_alt), dtype=float))  # Note: fwht output not normalized
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
        # a1 = @(x)(-np.floor(np.log2(x)))
        def a1(x):
            out = -np.floor(np.log2(x + np.finfo(float).eps))
            out[x == 0] = 0  # a1 is zero when x is zero
            return out

        # t1 = @(x)(2.^(-a1(x)))
        def t1(x):
            out = 2 ** (-a1(x))
            out[x == 0] = 0  # t1 is zero when x is zero
            return out

        # t2 = @(x)(2.^(-2*a1(x)))
        def t2(x):
            out = (2 ** (-2 * a1(x)))
            out[x == 0] = 0  # t2 is zero when x is zero
            return out

        s1 = lambda x: (1 - 2 * x)
        s2 = lambda x: (1 / 3 - 2 * (1 - x) * x)
        ts2 = lambda x: ((1 - 5 * t1(x)) / 2 - (a1(x) - 2) * x)
        ts3 = lambda x: ((1 - 43 * t2(x)) / 18 + (5 * t1(x) - 1) * x + (a1(x) - 2) * x ** 2)

        if order == 1:
            kernFunc = lambda x: (6 * ((1 / 6) - 2 ** (np.floor(np.log2(x + np.finfo(float).eps)) - 1)))
        elif order == 2:
            omega2_1D = lambda x: (s1(x) + ts2(x))
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
        self.fwht_copy_cf = _c_lib.fwht_copy
        self.fwht_copy_cf.argtypes = [
            ctypes.c_uint32,
            np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS')
        ]
        self.fwht_copy_cf.restype = None

        self.fwht_inplace_cf = _c_lib.fwht_inplace
        self.fwht_inplace_cf.argtypes = [
            ctypes.c_uint32,
            np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
        ]
        self.fwht_inplace_cf.restype = None

    def fwht_copy(self, n, src, dst):
        self.fwht_copy_cf(n, src, dst)

    def fwht_inplace(self, n, src):
        self.fwht_inplace_cf(n, src)


class CubBayesSobolG(CubBayesNetG): pass
class CubQMCBayesSobolG(CubBayesNetG): pass
class CubQMCBayesNetG(CubBayesNetG): pass
