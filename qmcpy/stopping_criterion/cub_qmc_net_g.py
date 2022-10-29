from ._cub_qmc_ld_g import _CubQMCLDG
from ..util import ParameterError
from ..discrete_distribution import DigitalNetB2
from ..true_measure import Gaussian, Uniform
from ..integrand import Keister, CustomFun, BoxIntegral
from numpy import *


class CubQMCNetG(_CubQMCLDG):
    r"""
    Quasi-Monte Carlo method using Sobol' cubature over the
    d-dimensional region to integrate within a specified generalized error
    tolerance with guarantees under Walsh-Fourier coefficients cone decay assumptions.

    >>> k = Keister(DigitalNetB2(2,seed=7))
    >>> sc = CubQMCNetG(k,abs_tol=.05)
    >>> solution,data = sc.integrate()
    >>> data
    LDTransformData (AccumulateData Object)
        solution        1.809
        comb_bound_low  1.804
        comb_bound_high 1.814
        comb_flags      1
        n_total         2^(10)
        n               2^(10)
        time_integrate  ...
    CubQMCNetG (StoppingCriterion Object)
        abs_tol         0.050
        rel_tol         0
        n_init          2^(10)
        n_max           2^(35)
    Keister (Integrand Object)
    Gaussian (TrueMeasure Object)
        mean            0
        covariance      2^(-1)
        decomp_type     PCA
    DigitalNetB2 (DiscreteDistribution Object)
        d               2^(1)
        dvec            [0 1]
        randomize       LMS_DS
        graycode        0
        entropy         7
        spawn_key       ()
    >>> dd = DigitalNetB2(3,seed=7)
    >>> g1 = CustomFun(Uniform(dd,0,2),lambda t: 10*t[:,0]-5*t[:,1]**2+t[:,2]**3)
    >>> cv1 = CustomFun(Uniform(dd,0,2),lambda t: t[:,0])
    >>> cv2 = CustomFun(Uniform(dd,0,2),lambda t: t[:,1]**2)
    >>> sc = CubQMCNetG(g1,abs_tol=1e-6,check_cone=True,
    ...     control_variates = [cv1,cv2],
    ...     control_variate_means = [1,4/3])
    >>> sol,data = sc.integrate()
    >>> sol
    array([5.33333333])
    >>> exactsol = 16/3
    >>> abs(sol-exactsol)<1e-6
    array([ True])
    >>> dnb2 = DigitalNetB2(3,seed=7)
    >>> f = BoxIntegral(dnb2, s=[-1,1])
    >>> abs_tol = 1e-3
    >>> sc = CubQMCNetG(f, abs_tol=abs_tol)
    >>> solution,data = sc.integrate()
    >>> solution
    array([1.18944142, 0.96064165])
    >>> sol3neg1 = -pi/4-1/2*log(2)+log(5+3*sqrt(3))
    >>> sol31 = sqrt(3)/4+1/2*log(2+sqrt(3))-pi/24
    >>> true_value = array([sol3neg1,sol31])
    >>> (abs(true_value-solution)<abs_tol).all()
    True
    >>> f2 = BoxIntegral(dnb2,s=[3,4])
    >>> sc = CubQMCNetG(f2,control_variates=f,control_variate_means=true_value,update_beta=True)
    >>> solution,data = sc.integrate()
    >>> solution
    array([1.10168119, 1.26661293])
    >>> data
    LDTransformData (AccumulateData Object)
        solution        [1.102 1.267]
        comb_bound_low  [1.099 1.262]
        comb_bound_high [1.104 1.271]
        comb_flags      [ True  True]
        n_total         2^(10)
        n               [1024. 1024.]
        time_integrate  ...
    CubQMCNetG (StoppingCriterion Object)
        abs_tol         0.010
        rel_tol         0
        n_init          2^(10)
        n_max           2^(35)
        cv              BoxIntegral (Integrand Object)
                           s               [-1  1]
        cv_mu           [1.19  0.961]
        update_beta     1
    BoxIntegral (Integrand Object)
        s               [3 4]
    Uniform (TrueMeasure Object)
        lower_bound     0
        upper_bound     1
    DigitalNetB2 (DiscreteDistribution Object)
        d               3
        dvec            [0 1 2]
        randomize       LMS_DS
        graycode        0
        entropy         7
        spawn_key       ()
    >>> cf = CustomFun(
    ...     true_measure = Uniform(DigitalNetB2(6,seed=7)),
    ...     g = lambda x,compute_flags=None: (2*arange(1,7)*x).reshape(-1,2,3),
    ...     rho = (2,3))
    >>> sol,data = CubQMCNetG(cf,abs_tol=1e-6).integrate()
    >>> data
    LDTransformData (AccumulateData Object)
        solution        [[1. 2. 3.]
                        [4. 5. 6.]]
        comb_bound_low  [[1. 2. 3.]
                        [4. 5. 6.]]
        comb_bound_high [[1. 2. 3.]
                        [4. 5. 6.]]
        comb_flags      [[ True  True  True]
                        [ True  True  True]]
        n_total         2^(13)
        n               [[2048. 1024. 1024.]
                        [8192. 4096. 2048.]]
        time_integrate  ...
    CubQMCNetG (StoppingCriterion Object)
        abs_tol         1.00e-06
        rel_tol         0
        n_init          2^(10)
        n_max           2^(35)
    CustomFun (Integrand Object)
    Uniform (TrueMeasure Object)
        lower_bound     0
        upper_bound     1
    DigitalNetB2 (DiscreteDistribution Object)
        d               6
        dvec            [0 1 2 3 4 5]
        randomize       LMS_DS
        graycode        0
        entropy         7
        spawn_key       ()

    Original Implementation:

        https://github.com/GailGithub/GAIL_Dev/blob/master/Algorithms/IntegrationExpectation/cubSobol_g.m

    References:

        [1] Fred J. Hickernell and Lluis Antoni Jimenez Rugama, 
        Reliable adaptive cubature using digital sequences, 2014. 
        Submitted for publication: arXiv:1410.8615.
        
        [2] Sou-Cheng T. Choi, Yuhan Ding, Fred J. Hickernell, Lan Jiang, Lluis Antoni Jimenez Rugama,
        Da Li, Jagadeeswaran Rathinavel, Xin Tong, Kan Zhang, Yizhi Zhang, and Xuan Zhou, 
        GAIL: Guaranteed Automatic Integration Library (Version 2.3) [MATLAB Software], 2019. 
        Available from http://gailgithub.github.io/GAIL_Dev/

    Guarantee:
        This algorithm computes the integral of real valued functions in $[0,1]^d$
        with a prescribed generalized error tolerance. The Fourier coefficients
        of the integrand are assumed to be absolutely convergent. If the
        algorithm terminates without warning messages, the output is given with
        guarantees under the assumption that the integrand lies inside a cone of
        functions. The guarantee is based on the decay rate of the Fourier
        coefficients. For integration over domains other than $[0,1]^d$, this cone
        condition applies to $f \circ \psi$ (the composition of the
        functions) where $\psi$ is the transformation function for $[0,1]^d$ to
        the desired region. For more details on how the cone is defined, please
        refer to the references below.
    """

    def __init__(self, integrand, abs_tol=1e-2, rel_tol=0., n_init=2.**10, n_max=2.**35,
        fudge=lambda m: 5.*2.**(-m), check_cone=False, 
        control_variates=[], control_variate_means=[], update_beta=False,
        error_fun = lambda sv,abs_tol,rel_tol: maximum(abs_tol,abs(sv)*rel_tol)):

        """
        Args:
            integrand (Integrand): an instance of Integrand
            abs_tol (ndarray): absolute error tolerance
            rel_tol (ndarray): relative error tolerance
            n_init (int): initial number of samples
            n_max (int): maximum number of samples
            fudge (function): positive function multiplying the finite
                              sum of Fast Fourier coefficients specified 
                              in the cone of functions
            check_cone (boolean): check if the function falls in the cone
            control_variates (list): list of integrand objects to be used as control variates. 
                Control variates are currently only compatible with single level problems. 
                The same discrete distribution instance must be used for the integrand and each of the control variates. 
            control_variate_means (list): list of means for each control variate
            update_beta (bool): update control variate beta coefficients at each iteration
            error_fun: function taking in the approximate solution vector, 
                absolute tolerance, and relative tolerance which returns the approximate error. 
                Default indicates integration until either absolute OR relative tolerance is satisfied.
        """
        super(CubQMCNetG,self).__init__(integrand,abs_tol,rel_tol,n_init,n_max,fudge,
            check_cone,control_variates,control_variate_means,update_beta,
            ptransform = 'none',
            coefv = lambda nl: ones(nl,dtype=float), 
            allowed_levels = ['single'],
            allowed_distribs = [DigitalNetB2],
            cast_complex = False,
            error_fun = error_fun)
        if (not self.discrete_distrib.randomize) or self.discrete_distrib.graycode:
            raise ParameterError("CubSobol_g requires distribution to have randomize=True and graycode=False.")


class CubQMCSobolG(CubQMCNetG): pass