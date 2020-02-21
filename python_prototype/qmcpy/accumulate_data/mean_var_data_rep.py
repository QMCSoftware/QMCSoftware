""" Definition for MeanVarDataRep, a concrete implementation of AccumulateData """

from ._accumulate_data import AccumulateData
from time import process_time
from numpy import array, finfo, float32, full, inf, nan, tile, zeros

EPS = finfo(float32).eps


class MeanVarDataRep(AccumulateData):
    """
    AccumulateData from Repeated Central Limit Stopping Criterion calculations.
    """

    parameters = ['replications','solution','sighat','n_total','confid_int']

    def __init__(self, stopping_criterion, integrand, n_init):
        """
        Initialize data instance

        Args:
            stopping_criterion (StoppingCriterion): a StoppingCriterion instance
            integrand (Integrand): an Integrand instance    
            n_init (int): initial number of samples
        """
        # Extract QMCPy objects
        self.stopping_criterion = stopping_criterion
        self.integrand = integrand
        self.measure = self.integrand.measure
        self.distribution = self.measure.distribution
        # Set Attributes
        self.replications = self.measure.distribution.replications
        self.muhat_r = zeros(self.replications)
        self.solution = nan
        self.muhat = inf  # sample mean
        self.sighat = inf # sample standard deviation
        self.t_eval = 0  # processing time for each integrand
        self.n = n_init  # currnet number of samples
        self.n_total = 0 # total number of samples
        self.confid_int = array([-inf, inf])  # confidence interval for solution
        super().__init__()

    def update_data(self):
        """ Update data """
        t_start = process_time()  # time integrand evaluation
        set_x = self.measure.gen_samples(n_min=self.n_total,n_max=self.n)
        for r in range(self.replications):
            y = self.integrand.f(set_x[r]).squeeze()
            previous_sum_y = self.muhat_r[r] * self.n_total
            self.muhat_r[r] = (y.sum() + previous_sum_y) / self.n  # updated integrand-replication mean
        self.muhat = self.muhat_r.mean()  # mean of replication streams means
        self.sighat = self.muhat_r.std()
        self.t_eval = max(process_time() - t_start, EPS)
        self.n_total = self.n  # updated the total evaluations
        # standard deviation of stream means
        self.solution = self.muhat.copy() # mean of integrand approximations
