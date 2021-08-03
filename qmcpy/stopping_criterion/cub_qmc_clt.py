from copy import deepcopy
from ._stopping_criterion import StoppingCriterion
from ..accumulate_data import MeanVarDataRep
from ..discrete_distribution._discrete_distribution import DiscreteDistribution
from ..discrete_distribution import Lattice
from ..true_measure import Gaussian
from ..integrand import Keister,BoxIntegral
from ..util import MaxSamplesWarning, NotYetImplemented, ParameterWarning, ParameterError
from numpy import *
from scipy.stats import norm
from time import time
import warnings


class CubQMCCLT(StoppingCriterion):
    """
    Stopping criterion based on Central Limit Theorem for multiple replications.
    
    >>> k = Keister(Lattice(seed=7))
    >>> sc = CubQMCCLT(k,abs_tol=.05)
    >>> solution,data = sc.integrate()
    >>> solution
    array([1.38049475])
    >>> data
    MeanVarDataRep (AccumulateData Object)
        solution        1.380
        error_bound     4.83e-04
        n_total         2^(12)
        n               2^(8)
        replications    2^(4)
        time_integrate  ...
    CubQMCCLT (StoppingCriterion Object)
        inflate         1.200
        alpha           0.010
        abs_tol         0.050
        rel_tol         0
        n_init          2^(8)
        n_max           2^(30)
    Keister (Integrand Object)
    Gaussian (TrueMeasure Object)
        mean            0
        covariance      2^(-1)
        decomp_type     pca
    Lattice (DiscreteDistribution Object)
        d               1
        randomize       1
        order           natural
        seed            7
        mimics          StdUniform
    >>> f = BoxIntegral(Lattice(3,seed=7), s=[-1,1])
    >>> abs_tol = 1e-3
    >>> sc = CubQMCCLT(f, abs_tol=abs_tol)
    >>> solution,data = sc.integrate()
    >>> solution
    array([1.19031076, 0.96093376])
    >>> data
    MeanVarDataRep (AccumulateData Object)
        solution        [1.19  0.961]
        error_bound     [0. 0.]
        n_total         2^(21)
        n               [131072.    512.]
        replications    2^(4)
        time_integrate  ...
    CubQMCCLT (StoppingCriterion Object)
        inflate         1.200
        alpha           0.010
        abs_tol         0.001
        rel_tol         0
        n_init          2^(8)
        n_max           2^(30)
    BoxIntegral (Integrand Object)
        s               [-1  1]
    Uniform (TrueMeasure Object)
        lower_bound     0
        upper_bound     1
    Lattice (DiscreteDistribution Object)
        d               3
        randomize       1
        order           natural
        seed            7
        mimics          StdUniform
    >>> sol3neg1 = -pi/4-1/2*log(2)+log(5+3*sqrt(3))
    >>> sol31 = sqrt(3)/4+1/2*log(2+sqrt(3))-pi/24
    >>> true_value = array([sol3neg1,sol31])
    >>> (abs(true_value-solution)<abs_tol).all()
    True
    """

    def __init__(self, integrand, abs_tol=1e-2, rel_tol=0., n_init=256., n_max=2**30,
        inflate=1.2, alpha=0.01, replications=16., 
        combine = lambda indv_sols: indv_sols,
        error_fun = lambda sv,abs_tol,rel_tol: maximum(abs_tol,abs(sv)*rel_tol),
        bound_fun = lambda phvl,phvh: (phvl,phvh,False),
        dependency = lambda flags_comb: flags_comb):
        """
        Args:
            integrand (Integrand): an instance of Integrand
            inflate (float): inflation factor when estimating variance
            alpha (float): significance level for confidence interval
            abs_tol (float): absolute error tolerance
            rel_tol (float): relative error tolerance
            n_max (int): maximum number of samples
            replications (int): number of replications
        """
        self.parameters = ['inflate','alpha','abs_tol','rel_tol','n_init','n_max']
        # Input Checks
        if log2(n_init) % 1 != 0:
            warning_s = ' n_init must be a power of 2. Using n_init = 32'
            warnings.warn(warning_s, ParameterWarning)
            n_init = 32
        # Set Attributes
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.n_init = float(n_init)
        self.n_max = float(n_max)
        self.alpha = float(alpha)
        self.z_star = -norm.ppf(self.alpha / 2)
        self.inflate = float(inflate)
        self.replications = replications
        self.combine = combine
        self.error_fun = error_fun
        self.bound_fun = bound_fun
        self.dependency = dependency
        # QMCPy Objs
        self.integrand = integrand
        self.true_measure = self.integrand.true_measure
        self.discrete_distrib = self.integrand.discrete_distrib
        # Verify Compliant Construction
        allowed_levels = ["single"]
        allowed_distribs = ["Lattice", "Sobol","Halton"]
        allow_vectorized_integrals = True
        super(CubQMCCLT,self).__init__(allowed_levels, allowed_distribs, allow_vectorized_integrals)
        if not self.discrete_distrib.randomize:
            raise ParameterError("CLTRep requires distribution to have randomize=True")
         
    def integrate(self):
        """ See abstract method. """
        # Construct AccumulateData Object to House Integration data
        self.data = MeanVarDataRep(self, self.integrand, self.true_measure, self.discrete_distrib, self.n_init, self.replications)
        t_start = time()
        while True:
            self.data.update_data()
            self.data.indv_error_bound = self.z_star * self.inflate * self.data.sighat / sqrt(self.data.replications)
            self.data.ci_low = self.data.solution_indv-self.data.indv_error_bound
            self.data.ci_high = self.data.solution_indv+self.data.indv_error_bound
            self.data.ci_comb_low,self.data.ci_comb_high,self.data.violated = self.bound_fun(self.data.ci_low,self.data.ci_high)
            self.data.solution_comb = self.combine(self.data.solution_indv)
            self.data.flags_comb = maximum(
                abs(self.data.ci_comb_high-self.data.solution_comb)-self.error_fun(self.data.ci_comb_high,self.abs_tol,self.rel_tol),
                abs(self.data.ci_comb_low-self.data.solution_comb)-self.error_fun(self.data.ci_comb_low,self.abs_tol,self.rel_tol))>=0
            self.data.flags_comb |= array([self.data.violated])
            self.data.flags_indv = self.dependency(self.data.flags_comb)
            if sum(self.data.flags_indv)==0:
                # sufficiently estimated
                break
            elif 2 * self.data.n_total > self.n_max:
                # doubling samples would go over n_max
                warning_s = """
                Alread generated %d samples.
                Trying to generate %d new samples would exceeds n_max = %d.
                No more samples will be generated.
                Note that error tolerances may not be satisfied""" \
                % (int(self.data.n_total), int(self.data.n_total), int(self.n_max))
                warnings.warn(warning_s, MaxSamplesWarning)
                break
            else:
                # double sample size
                self.data.nprev = where(self.data.flags_indv,self.data.n,self.data.nprev)
                self.data.n = where(self.data.flags_indv,2*self.data.n,self.data.n)
        self.data.time_integrate = time() - t_start
        self.data.solution = self.data.solution_comb
        return self.data.solution, self.data
    
    def set_tolerance(self, abs_tol=None, rel_tol=None):
        """
        See abstract method. 
        
        Args:
            abs_tol (float): absolute tolerance. Reset if supplied, ignored if not. 
            rel_tol (float): relative tolerance. Reset if supplied, ignored if not. 
        """
        if abs_tol != None: self.abs_tol = abs_tol
        if rel_tol != None: self.rel_tol = rel_tol