from ._cub_qmc_ld_g import _CubQMCLDG
from ..discrete_distribution import Lattice
from ..true_measure import Gaussian, Uniform
from ..integrand import Keister, BoxIntegral, CustomFun
from ..integrand.genz import Genz
from ..integrand.sensitivity_indices import SensitivityIndices
from ..util import fftbr,omega_fftbr,ParameterError
import numpy as np


class CubQMCLatticeG(_CubQMCLDG):
    r"""
    Stopping Criterion quasi-Monte Carlo method using rank-1 Lattices cubature over
    a d-dimensional region to integrate within a specified generalized error
    tolerance with guarantees under Fourier coefficients cone decay assumptions.
    
    Examples:
        >>> k = Keister(Lattice(seed=7))
        >>> sc = CubQMCLatticeG(k,abs_tol=1e-3,rel_tol=0,check_cone=True)
        >>> solution,data = sc.integrate()
        >>> solution
        array(1.38037385)
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
        CubQMCLatticeG (AbstractStoppingCriterion)
            abs_tol         0.001
            rel_tol         0
            n_init          2^(10)
            n_limit         2^(35)
        Keister (AbstractIntegrand)
        Gaussian (AbstractTrueMeasure)
            mean            0
            covariance      2^(-1)
            decomp_type     PCA
        Lattice (AbstractLDDiscreteDistribution)
            d               1
            replications    1
            randomize       SHIFT
            gen_vec_source  kuo.lattice-33002-1024-1048576.9125.txt
            order           NATURAL
            n_limit         2^(20)
            entropy         7
        
        Vector outputs
        
        >>> f = BoxIntegral(Lattice(3,seed=11),s=[-1,1])
        >>> abs_tol = 1e-3
        >>> sc = CubQMCLatticeG(f,abs_tol=abs_tol,rel_tol=0,check_cone=True)
        >>> solution,data = sc.integrate()
        >>> solution
        array([1.18947477, 0.96060862])
        >>> data
        AccumulateData (AccumulateData)
            solution        [1.189 0.961]
            comb_bound_low  [1.189 0.96 ]
            comb_bound_high [1.19  0.961]
            comb_bound_diff [0.001 0.001]
            comb_flags      [ True  True]
            n_total         2^(13)
            n               [8192 1024]
            time_integrate  ...
        CubQMCLatticeG (AbstractStoppingCriterion)
            abs_tol         0.001
            rel_tol         0
            n_init          2^(10)
            n_limit         2^(35)
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
            entropy         11
        >>> sol3neg1 = -np.pi/4-1/2*np.log(2)+np.log(5+3*np.sqrt(3))
        >>> sol31 = np.sqrt(3)/4+1/2*np.log(2+np.sqrt(3))-np.pi/24
        >>> true_value = np.array([sol3neg1,sol31])
        >>> assert (abs(true_value-solution)<abs_tol).all()

        Sensitivity indices 

        >>> function = Genz(Lattice(3,seed=7))
        >>> integrand = SensitivityIndices(function)
        >>> sc = CubQMCLatticeG(integrand,abs_tol=5e-4,rel_tol=0,check_cone=True)
        >>> solution,data = sc.integrate()
        >>> data
        AccumulateData (AccumulateData)
            solution        [[0.021 0.196 0.667]
                            [0.036 0.303 0.782]]
            comb_bound_low  [[0.02  0.196 0.667]
                            [0.035 0.303 0.781]]
            comb_bound_high [[0.021 0.196 0.667]
                            [0.036 0.303 0.782]]
            comb_bound_diff [[0.001 0.001 0.   ]
                            [0.001 0.001 0.001]]
            comb_flags      [[ True  True  True]
                            [ True  True  True]]
            n_total         2^(16)
            n               [[[16384 32768 65536]
                             [16384 32768 65536]
                             [16384 32768 65536]]
        <BLANKLINE>
                            [[ 2048 16384 32768]
                             [ 2048 16384 32768]
                             [ 2048 16384 32768]]]
            time_integrate  ...
        CubQMCLatticeG (AbstractStoppingCriterion)
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
        Lattice (AbstractLDDiscreteDistribution)
            d               6
            replications    1
            randomize       SHIFT
            gen_vec_source  kuo.lattice-33002-1024-1048576.9125.txt
            order           NATURAL
            n_limit         2^(20)
            entropy         7
    
    Original Implementation:

        https://github.com/GailGithub/GAIL_Dev/blob/master/Algorithms/IntegrationExpectation/cubLattice_g.m

    References:

        [1] Lluis Antoni Jimenez Rugama and Fred J. Hickernell, 
        "Adaptive multidimensional integration based on rank-1 lattices," 
        Monte Carlo and Quasi-Monte Carlo Methods: MCQMC, Leuven, Belgium, 
        April 2014 (R. Cools and D. Nuyens, eds.), Springer Proceedings in Mathematics 
        and Statistics, vol. 163, Springer-Verlag, Berlin, 2016, arXiv:1411.1966, pp. 407-422.
        
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
        fudge=lambda m: 5.*2.**(-m), check_cone=False, ptransform='Baker',
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
            error_fun: function taking in the approximate solution vector, 
                absolute tolerance, and relative tolerance which returns the approximate error. 
                Default indicates integration until either absolute OR relative tolerance is satisfied.
        """
        super(CubQMCLatticeG,self).__init__(integrand,abs_tol,rel_tol,n_init,n_max,fudge,check_cone,
            control_variates = [],
            control_variate_means = [],
            update_beta=False,
            ptransform = ptransform,
            ft = fftbr,
            omega = omega_fftbr,
            allowed_levels = ['single'],
            allowed_distribs = [Lattice],
            cast_complex = True,
            error_fun = error_fun)
        if self.discrete_distrib.order!='NATURAL':
            raise ParameterError("CubLattice_g requires Lattice with 'NATURAL' order")
