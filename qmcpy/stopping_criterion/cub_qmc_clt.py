from ._stopping_criterion import StoppingCriterion
from ..accumulate_data import MeanVarDataRep
from ..discrete_distribution._discrete_distribution import DiscreteDistribution
from ..discrete_distribution import Lattice
from ..true_measure import Gaussian
from ..integrand import Keister,XtoVectorizedPowers
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
        replications    2^(4)
        sighat          6.25e-04
        n               2^(8)
        n_total         2^(12)
        error_bound     4.83e-04
        confid_int      [[1.38 ]
                        [1.381]]
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
    >>> f = XtoVectorizedPowers(Lattice(2,seed=7), powers=[4,5])
    >>> sc = CubQMCCLT(f, abs_tol=1e-4)
    >>> solution,data = sc.integrate()
    >>> solution
    array([0.4  , 0.333])
    >>> data
    MeanVarDataRep (AccumulateData Object)
        solution        [0.4   0.333]
        replications    2^(4)
        sighat          [7.984e-05 7.984e-05]
        n               [4096. 4096.]
        n_total         2^(16)
        error_bound     [6.17e-05 6.17e-05]
        confid_int      [[0.4   0.333]
                        [0.4   0.333]]
        time_integrate  ...
    CubQMCCLT (StoppingCriterion Object)
        inflate         1.200
        alpha           0.010
        abs_tol         1.00e-04
        rel_tol         0
        n_init          2^(8)
        n_max           2^(30)
    XtoVectorizedPowers (Integrand Object)
    Uniform (TrueMeasure Object)
        lower_bound     0
        upper_bound     1
    Lattice (DiscreteDistribution Object)
        d               2^(1)
        randomize       1
        order           natural
        seed            7
        mimics          StdUniform

    """

    def __init__(self, integrand, abs_tol=1e-2, rel_tol=0., n_init=256., n_max=2**30,
                 inflate=1.2, alpha=0.01, replications=16.):
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
        self.abs_tol = float(abs_tol)
        self.rel_tol = float(rel_tol)
        self.n_init = float(n_init)
        self.n_max = float(n_max)
        self.alpha = float(alpha)
        self.z_star = -norm.ppf(self.alpha / 2)
        self.inflate = float(inflate)
        self.replications = replications
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
            self.data.error_bound = self.z_star * self.inflate * self.data.sighat / sqrt(self.data.replications)
            tol_up = maximum(self.abs_tol, abs(self.data.solution) * self.rel_tol)
            self.data.compute_flags = self.data.error_bound > tol_up
            if sum(self.data.compute_flags)==0:
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
                self.data.nprev = where(self.data.compute_flags,self.data.n,self.data.nprev)
                self.data.n = where(self.data.compute_flags,2*self.data.n,self.data.n)
        # CLT confidence interval
        self.data.confid_int = self.data.solution +  self.data.error_bound * array([[-1.],[1.]])
        self.data.time_integrate = time() - t_start
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