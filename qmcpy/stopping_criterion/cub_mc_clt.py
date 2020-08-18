from ._stopping_criterion import StoppingCriterion
from ..accumulate_data import MeanVarData
from ..discrete_distribution import IIDStdGaussian
from ..true_measure import Gaussian, BrownianMotion
from ..integrand import Keister, AsianOption
from ..util import MaxSamplesWarning
from numpy import *
from scipy.stats import norm
from time import time
import warnings


class CubMCCLT(StoppingCriterion):
    """
    Stopping criterion based on the Central Limit Theorem.
    
    >>> k = Keister(Gaussian(IIDStdGaussian(2,seed=7),covariance=1./2))
    >>> sc = CubMCCLT(k,abs_tol=.05)
    >>> solution,data = sc.integrate()
    >>> solution
    1.834674149189029
    >>> data
    Solution: 1.8347         
    Keister (Integrand Object)
    IIDStdGaussian (DiscreteDistribution Object)
        dimension       2^(1)
        seed            7
        mimics          StdGaussian
    Gaussian (TrueMeasure Object)
        mean            0
        covariance      2^(-1)
        decomp_type     pca
    CubMCCLT (StoppingCriterion Object)
        inflate         1.200
        alpha           0.010
        abs_tol         0.050
        rel_tol         0
        n_init          2^(10)
        n_max           10000000000
    MeanVarData (AccumulateData Object)
        levels          1
        solution        1.835
        n               5826
        n_total         6850
        error_bound     0.050
        confid_int      [1.785 1.885]
        time_integrate  ...
    >>> ac = AsianOption(
    ...     measure = BrownianMotion(IIDStdGaussian()),
    ...     multi_level_dimensions = [2,4,8])
    >>> sc = CubMCCLT(ac,abs_tol=.05)
    >>> solution,data = sc.integrate()
    """

    parameters = ['inflate','alpha','abs_tol','rel_tol','n_init','n_max']
    
    def __init__(self, integrand, abs_tol=1e-2, rel_tol=0., n_init=1024., n_max=1e10,
                 inflate=1.2, alpha=0.01):
        """
        Args:
            integrand (Integrand): an instance of Integrand
            inflate (float): inflation factor when estimating variance
            alpha (float): significance level for confidence interval
            abs_tol (float): absolute error tolerance
            rel_tol (float): relative error tolerance
            n_max (int): maximum number of samples
        """
        # Set Attributes
        self.abs_tol = float(abs_tol)
        self.rel_tol = float(rel_tol)
        self.n_init = float(n_init)
        self.n_max = float(n_max)
        self.alpha = float(alpha)
        self.inflate = float(inflate)
        self.integrand = integrand
        # Verify Compliant Construction
        distribution = integrand.measure.distribution
        allowed_levels = ['single','fixed-multi']
        allowed_distribs = ["IIDStdUniform", "IIDStdGaussian", "CustomIIDDistribution"]
        super(CubMCCLT,self).__init__(distribution, integrand, allowed_levels, allowed_distribs)

    def integrate(self):
        """ See abstract method. """
        # Construct AccumulateData Object to House Integration data
        self.data = MeanVarData(self, self.integrand, self.n_init)
        t_start = time()
        # Pilot Sample
        self.data.update_data()
        # use cost of function values to decide how to allocate
        temp_a = self.data.t_eval ** 0.5
        temp_b = (temp_a * self.data.sighat).sum()
        # samples for computation of the mean
        # n_mu_temp := n such that confidence intervals width and conficence will be satisfied
        tol_up = max(self.abs_tol, abs(self.data.solution) * self.rel_tol)
        z_star = -norm.ppf(self.alpha / 2.)
        n_mu_temp = ceil(temp_b * (self.data.sighat / temp_a) * \
                            (z_star * self.inflate / tol_up)**2)
        # n_mu := n_mu_temp adjusted for previous n
        self.data.n_mu = maximum(self.data.n, n_mu_temp)
        self.data.n += self.data.n_mu.astype(int)
        if self.data.n_total + self.data.n.sum() > self.n_max:
            # cannot generate this many new samples
            warning_s = """
            Alread generated %d samples.
            Trying to generate %d new samples, which would exceed n_max = %d.
            The number of new samples will be decrease proportionally for each integrand.
            Note that error tolerances may no longer be satisfied""" \
            % (int(self.data.n_total), int(self.data.n.sum()), int(self.n_max))
            warnings.warn(warning_s, MaxSamplesWarning)
            # decrease n proportionally for each integrand
            n_decease = self.data.n_total + self.data.n.sum() - self.n_max
            dec_prop = n_decease / self.data.n.sum()
            self.data.n = floor(self.data.n - self.data.n * dec_prop)
        # Final Sample
        self.data.update_data()
        # CLT confidence interval
        sigma_up = (self.data.sighat ** 2 / self.data.n_mu).sum(0) ** 0.5
        self.data.error_bound = z_star * self.inflate * sigma_up
        self.data.confid_int = self.data.solution + self.data.error_bound * array([-1, 1])
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