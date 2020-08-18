from ._stopping_criterion import StoppingCriterion
from ..accumulate_data import MeanVarDataRep
from ..discrete_distribution._discrete_distribution import DiscreteDistribution
from ..discrete_distribution import Lattice
from ..true_measure import Gaussian
from ..integrand import Keister
from ..util import MaxSamplesWarning, NotYetImplemented, ParameterWarning, ParameterError
from numpy import *
from scipy.stats import norm
from time import time
import warnings


class CubQMCCLT(StoppingCriterion):
    """
    Stopping criterion based on Central Limit Theorem for multiple replications.
    
    >>> k = Keister(Gaussian(Lattice(seed=7),covariance=1./2))
    >>> sc = CubQMCCLT(k,abs_tol=.05)
    >>> solution,data = sc.integrate()
    >>> solution
    1.3798619783658828
    >>> data
    Solution: 1.3799         
    Keister (Integrand Object)
    Lattice (DiscreteDistribution Object)
        dimension       1
        randomize       1
        seed            1092
        mimics          StdUniform
        backend         GAIL
    Gaussian (TrueMeasure Object)
        mean            0
        covariance      2^(-1)
        decomp_type     pca
    CubQMCCLT (StoppingCriterion Object)
        inflate         1.200
        alpha           0.010
        abs_tol         0.050
        rel_tol         0
        n_init          2^(8)
        n_max           2^(30)
    MeanVarDataRep (AccumulateData Object)
        replications    2^(4)
        solution        1.380
        sighat          0.001
        n_total         2^(12)
        error_bound     8.30e-04
        confid_int      [1.379 1.381]
        time_integrate  ...
    """

    parameters = ['inflate','alpha','abs_tol','rel_tol','n_init','n_max']

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
        self.integrand = integrand
        # DiscreteDistribution checks
        distribution = integrand.measure.distribution
        allowed_levels = ["single"]
        allowed_distribs = ["Lattice", "Sobol","Halton"]
        super(CubQMCCLT,self).__init__(distribution, integrand, allowed_levels, allowed_distribs)
        if not distribution.randomize:
            raise ParameterError("CLTRep requires distribution to have randomize=True")
         
    def integrate(self):
        """ See abstract method. """
        # Construct AccumulateData Object to House Integration data
        self.data = MeanVarDataRep(self, self.integrand, self.n_init, self.replications)
        t_start = time()
        while True:
            self.data.update_data()
            self.data.error_bound = self.z_star * self.inflate * self.data.sighat / sqrt(self.data.replications)
            tol_up = max(self.abs_tol, abs(self.data.solution) * self.rel_tol)
            if self.data.error_bound < tol_up:
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
                self.data.n_r *= 2
        # CLT confidence interval
        self.data.confid_int = self.data.solution +  self.data.error_bound * array([-1., 1.])
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