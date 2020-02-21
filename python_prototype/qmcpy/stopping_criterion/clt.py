""" Definition for CLT, a concrete implementation of StoppingCriterion """

from ._stopping_criterion import StoppingCriterion
from ..accumulate_data import MeanVarData
from ..discrete_distribution._discrete_distribution import DiscreteDistribution
from ..util import MaxSamplesWarning, ParameterError
import warnings
from numpy import array, ceil, floor, maximum
from scipy.stats import norm


class CLT(StoppingCriterion):
    """ Stopping criterion based on the Central Limit Theorem (CLT) """

    parameters = ['inflate','alpha','abs_tol','rel_tol','n_init','n_max']
    
    def __init__(self, distributions, inflate=1.2, alpha=0.01,
                 abs_tol=1e-2, rel_tol=0, n_init=1024, n_max=1e10):
        """
        Args:
            distributions (DiscreteDistribution): an instance of DiscreteDistribution
            inflate (float): inflation factor when estimating variance
            alpha (float): significance level for confidence interval
            abs_tol (float): absolute error tolerance
            rel_tol (float): relative error tolerance
            n_max (int): maximum number of samples
        """
        # Set Attributes
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.n_init = n_init
        self.n_max = n_max
        self.alpha = alpha
        self.inflate = inflate
        self.stage = "sigma"
        # Construct AccumulateData Object to House Integration data
        if isinstance(distributions,DiscreteDistribution): 
            levels = 1 # single level problem
        else: # list of DiscreteDistribution instances
            levels = len(distributions)
        self.data = MeanVarData(levels, n_init)
        # Verify Compliant Construction
        allowed_distribs = ["IIDStdUniform", "IIDStdGaussian"]
        super().__init__(distributions, allowed_distribs)

    def stop_yet(self):
        """ Determine when to stop """
        if self.stage == "sigma":
            temp_a = self.data.t_eval ** 0.5
            # use cost of function values to decide how to allocate
            temp_b = (temp_a * self.data.sighat).sum(0)
            # samples for computation of the mean
            # n_mu_temp := n such that confidence intervals width and conficence will be satisfied
            tol_up = max(self.abs_tol, abs(self.data.solution) * self.rel_tol)
            z_star = -norm.ppf(self.alpha / 2)
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
            self.stage = "mu"  # compute sample mean next
        elif self.stage == "mu":
            # CLT confidence interval
            z_star = -norm.ppf(self.alpha / 2)
            sigma_up = (self.data.sighat ** 2 / self.data.n_mu).sum(0) ** 0.5
            err_bar = z_star * self.inflate * sigma_up
            self.data.confid_int = self.data.solution + err_bar * array([-1, 1])
            self.stage = "done"  # finished with computation
