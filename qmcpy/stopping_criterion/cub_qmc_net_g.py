from ._cub_qmc_ld_g import _CubQMCLDG
from ..util import fwht,omega_fwht,ParameterError
from ..discrete_distribution import DigitalNetB2
from ..true_measure import Gaussian, Uniform
from ..integrand import Keister, BoxIntegral, CustomFun
from ..integrand.genz import Genz
from ..integrand.sensitivity_indices import SensitivityIndices
import numpy as np


class CubQMCNetG(_CubQMCLDG):
    r"""
    Quasi-Monte Carlo method using Sobol' cubature over the
    d-dimensional region to integrate within a specified generalized error
    tolerance with guarantees under Walsh-Fourier coefficients cone decay assumptions.

    Examples:
        >>> k = Keister(DigitalNetB2(seed=7))
        >>> sc = CubQMCNetG(k,abs_tol=1e-3,rel_tol=0,check_cone=True)
        >>> solution,data = sc.integrate()
        >>> solution
        array(1.38038574)
        >>> data
        AccumulateData (AccumulateData)
            solution        1.380
            comb_bound_low  1.380
            comb_bound_high 1.381
            comb_bound_diff 0.001
            comb_flags      1
            n_total         2^(11)
            n               2^(11)
            time_integrate  ...
        CubQMCNetG (AbstractStoppingCriterion)
            abs_tol         0.001
            rel_tol         0
            n_init          2^(10)
            n_limit         2^(35)
        Keister (AbstractIntegrand)
        Gaussian (AbstractTrueMeasure)
            mean            0
            covariance      2^(-1)
            decomp_type     PCA
        DigitalNetB2 (AbstractLDDiscreteDistribution)
            d               1
            replications    1
            randomize       LMS_DS
            gen_mats_source joe_kuo.6.21201.txt
            order           NATURAL
            t               63
            alpha           1
            n_limit         2^(32)
            entropy         7
        
        Vector outputs
        
        >>> f = BoxIntegral(DigitalNetB2(3,seed=7),s=[-1,1])
        >>> abs_tol = 1e-3
        >>> sc = CubQMCNetG(f,abs_tol=abs_tol,rel_tol=0,check_cone=True)
        >>> solution,data = sc.integrate()
        >>> solution
        array([1.18965698, 0.96061461])
        >>> data
        AccumulateData (AccumulateData)
            solution        [1.19  0.961]
            comb_bound_low  [1.189 0.96 ]
            comb_bound_high [1.19  0.961]
            comb_bound_diff [0.001 0.001]
            comb_flags      [ True  True]
            n_total         2^(14)
            n               [16384  1024]
            time_integrate  ...
        CubQMCNetG (AbstractStoppingCriterion)
            abs_tol         0.001
            rel_tol         0
            n_init          2^(10)
            n_limit         2^(35)
        BoxIntegral (AbstractIntegrand)
            s               [-1  1]
        Uniform (AbstractTrueMeasure)
            lower_bound     0
            upper_bound     1
        DigitalNetB2 (AbstractLDDiscreteDistribution)
            d               3
            replications    1
            randomize       LMS_DS
            gen_mats_source joe_kuo.6.21201.txt
            order           NATURAL
            t               63
            alpha           1
            n_limit         2^(32)
            entropy         7
        >>> sol3neg1 = -np.pi/4-1/2*np.log(2)+np.log(5+3*np.sqrt(3))
        >>> sol31 = np.sqrt(3)/4+1/2*np.log(2+np.sqrt(3))-np.pi/24
        >>> true_value = np.array([sol3neg1,sol31])
        >>> assert (abs(true_value-solution)<abs_tol).all()

        Sensitivity indices 

        >>> function = Genz(DigitalNetB2(3,seed=7))
        >>> integrand = SensitivityIndices(function)
        >>> sc = CubQMCNetG(integrand,abs_tol=5e-4,rel_tol=0,check_cone=True)
        >>> solution,data = sc.integrate()
        >>> data
        AccumulateData (AccumulateData)
            solution        [[0.02  0.196 0.667]
                            [0.036 0.303 0.782]]
            comb_bound_low  [[0.019 0.195 0.667]
                            [0.035 0.303 0.781]]
            comb_bound_high [[0.02  0.196 0.667]
                            [0.036 0.303 0.782]]
            comb_bound_diff [[0.001 0.    0.001]
                            [0.001 0.001 0.001]]
            comb_flags      [[ True  True  True]
                            [ True  True  True]]
            n_total         2^(16)
            n               [[[16384 65536 65536]
                             [16384 65536 65536]
                             [16384 65536 65536]]
        <BLANKLINE>
                            [[ 2048 16384 32768]
                             [ 2048 16384 32768]
                             [ 2048 16384 32768]]]
            time_integrate  ...
        CubQMCNetG (AbstractStoppingCriterion)
            abs_tol         5.00e-04
            rel_tol         0
            n_init          2^(10)
            n_limit         2^(35)
        SensitivityIndices (AbstractIntegrand)
            indices         [[ True False False]
                            [False  True False]
                            [False False  True]]
        Uniform (AbstractTrueMeasure)
            lower_bound     0
            upper_bound     1
        DigitalNetB2 (AbstractLDDiscreteDistribution)
            d               6
            replications    1
            randomize       LMS_DS
            gen_mats_source joe_kuo.6.21201.txt
            order           NATURAL
            t               63
            alpha           1
            n_limit         2^(32)
            entropy         7
    
        Control Variates

        >>> dnb2 = DigitalNetB2(dimension=4,seed=7)
        >>> integrand = CustomFun(
        ...     true_measure = Uniform(dnb2),
        ...     g = lambda t,compute_flags: np.stack([
        ...         1*t[...,0]+2*t[...,0]**2+3*t[...,0]**3,
        ...         2*t[...,1]+3*t[...,1]**2+4*t[...,1]**3,
        ...         3*t[...,2]+4*t[...,2]**2+5*t[...,2]**3]),
        ...     dimension_indv = (3,))
        >>> control_variates = [
        ...     CustomFun(
        ...         true_measure = Uniform(dnb2),
        ...         g = lambda t,compute_flags: np.stack([t[...,0],t[...,1],t[...,2]],axis=0),
        ...         dimension_indv = (3,)),
        ...     CustomFun(
        ...         true_measure = Uniform(dnb2),
        ...         g = lambda t,compute_flags: np.stack([t[...,0]**2,t[...,1]**2,t[...,2]**2],axis=0),
        ...         dimension_indv = (3,))]
        >>> control_variate_means = np.array([[1/2,1/2,1/2],[1/3,1/3,1/3]])
        >>> true_value = np.array([23/12,3,49/12])
        >>> abs_tol = 1e-6
        >>> sc = CubQMCNetG(integrand,abs_tol=abs_tol,rel_tol=0,control_variates=control_variates,control_variate_means=control_variate_means,update_beta=False)
        >>> solution,data = sc.integrate()
        >>> solution
        array([1.91666667, 3.        , 4.08333333])
        >>> data.n
        array([4096, 8192, 8192])
        >>> assert (np.abs(true_value-solution)<abs_tol).all()
        >>> sc = CubQMCNetG(integrand,abs_tol=abs_tol,rel_tol=0,control_variates=control_variates,control_variate_means=control_variate_means,update_beta=True)
        >>> solution,data = sc.integrate()
        >>> solution
        array([1.91666667, 3.        , 4.08333333])
        >>> data.n
        array([ 8192, 16384, 16384])
        >>> assert (np.abs(true_value-solution)<abs_tol).all()

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
        error_fun = lambda sv,abs_tol,rel_tol: np.maximum(abs_tol,abs(sv)*rel_tol)):

        """
        Args:
            integrand (AbstractIntegrand): an instance of AbstractIntegrand
            abs_tol (np.ndarray): absolute error tolerance
            rel_tol (np.ndarray): relative error tolerance
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
            ft = fwht,
            omega = omega_fwht,
            allowed_levels = ['single'],
            allowed_distribs = [DigitalNetB2],
            cast_complex = False,
            error_fun = error_fun)
        if self.discrete_distrib.order!='NATURAL':
            raise ParameterError("CubQMCNet_g requires DigitalNetB2 with 'NATURAL' order")

class CubQMCSobolG(CubQMCNetG): pass
