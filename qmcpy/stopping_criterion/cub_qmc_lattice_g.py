from ._cub_qmc_ld_g import _CubQMCLDG
from ..discrete_distribution import Lattice
from ..true_measure import Gaussian
from ..integrand import Keister
from ..util import ParameterError
from numpy import *


class CubQMCLatticeG(_CubQMCLDG):
    """
    Stopping Criterion quasi-Monte Carlo method using rank-1 Lattices cubature over
    a d-dimensional region to integrate within a specified generalized error
    tolerance with guarantees under Fourier coefficients cone decay assumptions.
    
    >>> k = Keister(Lattice(2,seed=7))
    >>> sc = CubQMCLatticeG(k,abs_tol=.05)
    >>> solution,data = sc.integrate()
    >>> solution
    1.806...
    >>> data
    Solution: 1.8068         
    Keister (Integrand Object)
    Lattice (DiscreteDistribution Object)
        d               2^(1)
        randomize       1
        order           natural
        seed            7
        mimics          StdUniform
    Gaussian (TrueMeasure Object)
        mean            0
        covariance      2^(-1)
        decomp_type     pca
    CubQMCLatticeG (StoppingCriterion Object)
        abs_tol         0.050
        rel_tol         0
        n_init          2^(10)
        n_max           2^(35)
    LDTransformData (AccumulateData Object)
        n_total         2^(10)
        solution        1.807
        error_bound     0.005
        time_integrate  ...
    
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

    Guarantee
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
        fudge=lambda m: 5.*2.**(-m), check_cone=False, ptransform='Baker'):
        """
        Args:
            integrand (Integrand): an instance of Integrand
            abs_tol (float): absolute error tolerance
            rel_tol (float): relative error tolerance
            n_init (int): initial number of samples
            n_max (int): maximum number of samples
            fudge (function): positive function multiplying the finite
                              sum of Fast Fourier coefficients specified 
                              in the cone of functions
            check_cone (boolean): check if the function falls in the cone
        """
        super(CubQMCLatticeG,self).__init__(integrand,abs_tol,rel_tol,n_init,n_max,fudge,check_cone,
            control_variates = [],
            control_variate_means = [],
            update_beta=False,
            ptransform = ptransform,
            coefv = lambda nl: exp(-2*pi*1j*arange(nl)/(2*nl)), 
            allowed_levels = ['single'],
            allowed_distribs = ["Lattice"],
            cast_complex = True)
        if not self.discrete_distrib.randomize:
            raise ParameterError("CubLattice_g requires distribution to have randomize=True")
        if self.discrete_distrib.order != 'natural':
            raise ParameterError("CubLattice_g requires Lattice with 'natural' order")
