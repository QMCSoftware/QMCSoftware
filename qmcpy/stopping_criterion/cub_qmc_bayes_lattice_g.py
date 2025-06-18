from ._cub_bayes_ld_g import _CubBayesLDG
from ..discrete_distribution import Lattice
from ..integrand import Keister,BoxIntegral,Genz,SensitivityIndices
from ..util import fftbr,omega_fftbr,ParameterError#, ParameterWarning #MaxSamplesWarning,
#from math import factorial
import numpy as np
#from time import time
#import warnings


class CubBayesLatticeG(_CubBayesLDG):
    r"""
    Stopping criterion for Bayesian Cubature using rank-1 Lattice sequence with guaranteed
    accuracy over a d-dimensional region to integrate within a specified generalized error
    tolerance with guarantees under Bayesian assumptions.

    >>> k = Keister(Lattice(2, seed=123456789))
    >>> sc = CubBayesLatticeG(k,abs_tol=1e-4)
    >>> solution,data = sc.integrate()
    >>> data
    AccumulateData (AccumulateData)
        solution        1.808
        comb_bound_low  1.808
        comb_bound_high 1.808
        comb_bound_diff 3.60e-05
        comb_flags      1
        n_total         2^(10)
        n               2^(10)
        time_integrate  ...
    CubBayesLatticeG (AbstractStoppingCriterion)
        abs_tol         1.00e-04
        rel_tol         0
        n_init          2^(8)
        n_limit         2^(22)
        order           2^(1)
    Keister (AbstractIntegrand)
    Gaussian (AbstractTrueMeasure)
        mean            0
        covariance      2^(-1)
        decomp_type     PCA
    Lattice (AbstractLDDiscreteDistribution)
        d               2^(1)
        replications    1
        randomize       SHIFT
        gen_vec_source  kuo.lattice-33002-1024-1048576.9125.txt
        order           NATURAL
        n_limit         2^(20)
        entropy         123456789

        Vector outputs
        
        >>> f = BoxIntegral(Lattice(3,seed=7),s=[-1,1])
        >>> abs_tol = 1e-2
        >>> sc = CubBayesLatticeG(f,abs_tol=abs_tol,rel_tol=0)
        >>> solution,data = sc.integrate()
        >>> solution
        array([1.18837601, 0.95984299])
        >>> data
        AccumulateData (AccumulateData)
            solution        [1.188 0.96 ]
            comb_bound_low  [1.183 0.95 ]
            comb_bound_high [1.194 0.969]
            comb_bound_diff [0.011 0.019]
            comb_flags      [ True  True]
            n_total         2^(9)
            n               [512 256]
            time_integrate  ...
        CubBayesLatticeG (AbstractStoppingCriterion)
            abs_tol         0.010
            rel_tol         0
            n_init          2^(8)
            n_limit         2^(22)
            order           2^(1)
        BoxIntegral (AbstractIntegrand)
            s               [-1  1]
        Uniform (AbstractTrueMeasure)
            lower_bound     0
            upper_bound     1
        Lattice (AbstractLDDiscreteDistribution)
            d               3
            replications    1
            randomize       SHIFT
            gen_vec_source  kuo.lattice-33002-1024-1048576.9125.txt
            order           NATURAL
            n_limit         2^(20)
            entropy         7
        >>> sol3neg1 = -np.pi/4-1/2*np.log(2)+np.log(5+3*np.sqrt(3))
        >>> sol31 = np.sqrt(3)/4+1/2*np.log(2+np.sqrt(3))-np.pi/24
        >>> true_value = np.array([sol3neg1,sol31])
        >>> assert (abs(true_value-solution)<abs_tol).all()

        Sensitivity indices 

        >>> function = Genz(Lattice(3,seed=7))
        >>> integrand = SensitivityIndices(function)
        >>> sc = CubBayesLatticeG(integrand,abs_tol=5e-2,rel_tol=0)
        >>> solution,data = sc.integrate()
        >>> data
        AccumulateData (AccumulateData)
            solution        [[0.057 0.131 0.269]
                             [0.386 0.523 0.741]]
            comb_bound_low  [[0.048 0.12  0.256]
                             [0.377 0.511 0.721]]
            comb_bound_high [[0.066 0.141 0.283]
                             [0.395 0.535 0.762]]
            comb_bound_diff [[0.018 0.021 0.027]
                             [0.018 0.024 0.04 ]]
            comb_flags      [[ True  True  True]
                             [ True  True  True]]
            n_total         2^(10)
            n               [[[1024 1024 1024]
                              [1024 1024 1024]
                              [1024 1024 1024]]
        <BLANKLINE>
                             [[1024 1024 1024]
                              [1024 1024 1024]
                              [1024 1024 1024]]]
            time_integrate  ...
        CubBayesLatticeG (AbstractStoppingCriterion)
            abs_tol         0.050
            rel_tol         0
            n_init          2^(8)
            n_limit         2^(22)
            order           2^(1)
        SensitivityIndices (AbstractIntegrand)
            indices         [[ True False False]
                             [False  True False]
                             [False False  True]]
        Uniform (AbstractTrueMeasure)
            lower_bound     0
            upper_bound     1
        Lattice (AbstractLDDiscreteDistribution)
            d               6
            replications    1
            randomize       SHIFT
            gen_vec_source  kuo.lattice-33002-1024-1048576.9125.txt
            order           NATURAL
            n_limit         2^(20)
            entropy         7

    Adapted from `GAIL cubBayesLattice_g <https://github.com/GailGithub/GAIL_Dev/blob/master/Algorithms/IntegrationExpectation/cubBayesLattice_g.m>`_.

    Guarantees:
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
        [1] Jagadeeswaran Rathinavel and Fred J. Hickernell, Fast automatic Bayesian cubature using lattice sampling.
        Stat Comput 29, 1215-1229 (2019). Available from `Springer <https://doi.org/10.1007/s11222-019-09895-9>`_.

        [2] Sou-Cheng T. Choi, Yuhan Ding, Fred J. Hickernell, Lan Jiang, Lluis Antoni Jimenez Rugama,
        Da Li, Jagadeeswaran Rathinavel, Xin Tong, Kan Zhang, Yizhi Zhang, and Xuan Zhou,
        GAIL: Guaranteed Automatic Integration Library (Version 2.3) [MATLAB Software], 2019.
        Available from `GAIL <http://gailgithub.github.io/GAIL_Dev/>`_.
    """

    def __init__(self, integrand, abs_tol=1e-2, rel_tol=0,
                 n_init=2**8, n_limit=2**22, alpha=0.01, ptransform='C1sin',
                 error_fun="EITHER", 
                 errbd_type="MLE", order=2,):
        r"""
        Args:
            integrand (AbstractIntegrand): An AbstractIntegrand.
            abs_tol (np.ndarray): Absolute error tolerance.
            rel_tol (np.ndarray): Relative error tolerance.
            n_init (int): Initial number of samples. 
            n_limit (int): Maximum number of samples.
            alpha (np.ndarray): Uncertainty level in $(0,1)$. 
            error_fun (callable): Function mapping the approximate solution, absolute error tolerance, and relative error tolerance to the current error bound.

                - The default $(\hat{\boldsymbol{\mu}},\varepsilon_\mathrm{abs},\varepsilon_\mathrm{rel}) \mapsto \max\{\varepsilon_\mathrm{abs},\lvert \hat{\boldsymbol{\mu}} \rvert \varepsilon_\mathrm{rel}\}$ 
                    means the approximation error must be below either the absolue error *or* relative error.
                - Setting to $(\hat{\boldsymbol{\mu}},\varepsilon_\mathrm{abs},\varepsilon_\mathrm{rel}) \mapsto \min\{\varepsilon_\mathrm{abs},\lvert \hat{\boldsymbol{\mu}} \rvert \varepsilon_\mathrm{rel}\}$ 
                    means the approximation error must be below either the absolue error *and* relative error. 
            ptransform (str): periodization transform applied to the integrand
            errbd_type (str): MLE, GCV, or FULL
            order (int): Bernoulli kernel's order. If zero, choose order automatically
        """
        super(CubBayesLatticeG, self).__init__(integrand, ft=fftbr, omega=omega_fftbr,
                                               ptransform=ptransform,
                                               allowed_distribs=[Lattice],
                                               kernel=self._shift_inv_kernel,
                                               abs_tol=abs_tol, rel_tol=rel_tol,
                                               n_init=n_init, n_limit=n_limit, alpha=alpha, error_fun=error_fun, errbd_type=errbd_type)
        self.order = order  # Bernoulli kernel's order. If zero, choose order automatically
        # private properties
        # full_Bayes - Full Bayes - assumes m and s^2 as hyperparameters
        # GCV - Generalized cross validation
        self.kernType = 1  # Type-1: Bernoulli polynomial based algebraic convergence, Type-2: Truncated series
        self._xfullundtype = float
        if self.discrete_distrib.order!='NATURAL':
            raise ParameterError("CubLattice_g requires Lattice with 'NATURAL' order")

    # Shift invariant kernel
    # C1 : first row of the covariance matrix
    # Lambda : eigen values of the covariance matrix
    # Lambda_ring = fft(C1 - 1)

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
            (vec_C1m1, C1_alt) = self.kernel_t(theta * const_mult, kernel_func(xun))

            lambda_factor = max(abs(vec_C1m1))
            C1_alt = C1_alt / lambda_factor
            vec_C1m1 = vec_C1m1 / lambda_factor
            # eigenvalues must be real : Symmetric pos definite Kernel
            vec_lambda_ring = np.real(fftbr(vec_C1m1)*np.sqrt(vec_C1m1.shape[-1]))

            vec_lambda = vec_lambda_ring.copy()
            vec_lambda[0] = vec_lambda_ring[0] + len(vec_lambda_ring) / lambda_factor

            if debug_enable:
                # eigenvalues must be real : Symmetric pos definite Kernel
                vec_lambda_direct = np.real(fftbr(C1_alt)*np.sqrt(C1_alt.shape[-1]))  # Note: fft output unnormalized
                if sum(abs(vec_lambda_direct - vec_lambda)) > 1:
                    print('Possible error: check vec_lambda_ring computation')
        else:
            # direct approach to compute first row of the kernel Gram matrix
            vec_C1 = np.prod(1 + theta * const_mult * kernel_func(xun), 2)
            # matlab's builtin fft is much faster and accurate
            # eigenvalues must be real : Symmetric pos definite Kernel
            vec_lambda = np.real(fftbr(vec_C1)*np.sqrt(vec_C1.shape[-1]))
            vec_lambda_ring = 0
            lambda_factor = 1

        return vec_lambda, vec_lambda_ring, lambda_factor
