"""
Stopping criterion based on the Centeral Limit Theorem
"""
from numpy import array, ceil, maximum, minimum, tile, zeros
from scipy.stats import norm

from . import StoppingCriterion
from ..accum_data.mean_var_data import MeanVarData


class CLTStopping(StoppingCriterion):
    """ Stopping criterion based on the Centeral Limit Theorem (CLT)

    Attributes:


    """

    def __init__(self, distrib_obj, inflate=1.2, alpha=0.01, abs_tol=1e-2,
                 rel_tol=0, n_init=1024, n_max=1e8):
        """
        Args:
            distrib_obj: an instance of DiscreteDistribution
            inflate: inflation factor when estimating variance
            alpha: significance level for confidence interval
            abs_tol: absolute error tolerance
            rel_tol: relative error tolerance
            n_init: initial number of samples
            n_max: maximum number of samples
        """
        allowed_distribs = ["IIDDistribution"] # supported distributions
        super().__init__(distrib_obj, allowed_distribs, abs_tol, rel_tol, n_init, n_max)
        self.inflate = inflate  # inflation factor
        self.alpha = alpha # uncertainty level
        self.stage = 'sigma'  # compute standard deviation next
        # Construct Data Object
        n_integrands = len(distrib_obj)
        self.data_obj = MeanVarData(n_integrands) # house integration data
        self.data_obj.n_mu = zeros(n_integrands)  # number of samples used to compute the sample mean
        self.data_obj.n_prev = zeros(n_integrands)  # initialize data object
        self.data_obj.n_next = tile(self.n_init, n_integrands) # initialize as n_init

    def stop_yet(self):
        """
        Determine when to stop

        """
        if self.stage == 'sigma':
            self.data_obj.n_prev = self.data_obj.n_next  # update place in the sequence
            temp_a = (self.data_obj.t_eval)**.5  # use cost of function values to decide how to allocate
            temp_b = (temp_a * self.data_obj.sighat).sum(0)  # samples for computation of the mean
            # n_mu_temp := n such that confidence intervals width and conficence will be satisfied
            n_mu_temp = ceil(temp_b*(-norm.ppf(self.alpha/2)*self.inflate /
                             max(self.abs_tol, self.data_obj.solution*self.rel_tol))**2
                             * (self.data_obj.sighat/self.data_obj.t_eval**.5))
            # n_mu := n_mu_temp adjusted for n_prev
            self.data_obj.n_mu = minimum(maximum(self.data_obj.n_next, n_mu_temp), self.n_max-self.data_obj.n_prev)
            self.data_obj.n_next = self.data_obj.n_mu + self.data_obj.n_prev # set next_n to n_mu_temp
            self.stage = 'mu'  # compute sample mean next
        elif self.stage == 'mu':
            self.data_obj.solution = self.data_obj.muhat.sum() # mean of integrand means
            self.data_obj.n_samples_total = self.data_obj.n_next
            err_bar = -norm.ppf(self.alpha/2) * self.inflate * (self.data_obj.sighat**2 / self.data_obj.n_mu).sum(0)**.5
            self.data_obj.confid_int = self.data_obj.solution + err_bar * array([-1, 1]) # CLT confidence interval
            self.stage = 'done'  # finished with computation
