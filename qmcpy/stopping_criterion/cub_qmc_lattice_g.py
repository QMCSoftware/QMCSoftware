from ._cub_qmc_ld_g import _CubQMCLDG
from ..discrete_distribution import Lattice
from ..true_measure import Gaussian, Uniform
from ..integrand import Keister, BoxIntegral, CustomFun
from ..util import ParameterError
from numpy import *


class CubQMCLatticeG(_CubQMCLDG):
    r"""
    Stopping Criterion quasi-Monte Carlo method using rank-1 Lattices cubature over
    a d-dimensional region to integrate within a specified generalized error
    tolerance with guarantees under Fourier coefficients cone decay assumptions.
    
    >>> k = Keister(Lattice(2,seed=7))
    >>> sc = CubQMCLatticeG(k,abs_tol=.05)
    >>> solution,data = sc.integrate()
    >>> data
    LDTransformData (AccumulateData Object)
        solution        1.810
        comb_bound_low  1.806
        comb_bound_high 1.815
        comb_flags      1
        n_total         2^(10)
        n               2^(10)
        time_integrate  ...
    CubQMCLatticeG (StoppingCriterion Object)
        abs_tol         0.050
        rel_tol         0
        n_init          2^(10)
        n_max           2^(35)
    Keister (Integrand Object)
    Gaussian (TrueMeasure Object)
        mean            0
        covariance      2^(-1)
        decomp_type     PCA
    Lattice (DiscreteDistribution Object)
        d               2^(1)
        dvec            [0 1]
        randomize       1
        order           natural
        entropy         7
        spawn_key       ()
    >>> f = BoxIntegral(Lattice(3,seed=7), s=[-1,1])
    >>> abs_tol = 1e-3
    >>> sc = CubQMCLatticeG(f, abs_tol=abs_tol)
    >>> solution,data = sc.integrate()
    >>> solution
    array([1.18954582, 0.96056304])
    >>> sol3neg1 = -pi/4-1/2*log(2)+log(5+3*sqrt(3))
    >>> sol31 = sqrt(3)/4+1/2*log(2+sqrt(3))-pi/24
    >>> true_value = array([sol3neg1,sol31])
    >>> (abs(true_value-solution)<abs_tol).all()
    True
    >>> cf = CustomFun(
    ...     true_measure = Uniform(Lattice(6,seed=7)),
    ...     g = lambda x,compute_flags=None: (2*arange(1,7)*x).reshape(-1,2,3),
    ...     rho = (2,3))
    >>> sol,data = CubQMCLatticeG(cf,abs_tol=1e-6).integrate()
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
        n_total         2^(15)
        n               [[ 8192. 16384. 16384.]
                        [16384. 32768. 32768.]]
        time_integrate  ...
    CubQMCLatticeG (StoppingCriterion Object)
        abs_tol         1.00e-06
        rel_tol         0
        n_init          2^(10)
        n_max           2^(35)
    CustomFun (Integrand Object)
    Uniform (TrueMeasure Object)
        lower_bound     0
        upper_bound     1
    Lattice (DiscreteDistribution Object)
        d               6
        dvec            [0 1 2 3 4 5]
        randomize       1
        order           natural
        entropy         7
        spawn_key       ()
    
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
            error_fun: function taking in the approximate solution vector, 
                absolute tolerance, and relative tolerance which returns the approximate error. 
                Default indicates integration until either absolute OR relative tolerance is satisfied.
        """
        super(CubQMCLatticeG,self).__init__(integrand,abs_tol,rel_tol,n_init,n_max,fudge,check_cone,
            control_variates = [],
            control_variate_means = [],
            update_beta=False,
            ptransform = ptransform,
            coefv = lambda nl: exp(-2*pi*1j*arange(nl)/(2*nl)), 
            allowed_levels = ['single'],
            allowed_distribs = [Lattice],
            cast_complex = True,
            error_fun = error_fun)
        if not self.discrete_distrib.randomize:
            raise ParameterError("CubLattice_g requires distribution to have randomize=True")
        if self.discrete_distrib.order != 'natural':
            raise ParameterError("CubLattice_g requires Lattice with 'natural' order")
