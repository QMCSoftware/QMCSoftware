""" Definition for MeanVarDataRep, a concrete implementation of AccumData """

from ._accum_data import AccumData
from time import process_time
from numpy import array, finfo, float32, full, inf, nan, tile, zeros

EPS = finfo(float32).eps


class MeanVarDataRep(AccumData):
    """
    Data from Repeated Central Limit Stopping Criterion calculations.
    """

    parameters = ['replications','solution','sighat','n_total','confid_int']

    def __init__(self, n_init, replications):
        """
        Initialize data instance

        Args:
            n_init (int): initial number of samples
            replications (int): number of random nxm matrices to generate
        """
        self.replications = replications  # Number of random nxm matrices to generate
        self.muhat_r = zeros(self.replications)
        self.solution = nan
        self.muhat = inf  # sample mean
        self.sighat = inf # sample standard deviation
        self.t_eval = 0  # processing time for each integrand
        self.n = n_init  # currnet number of samples
        self.n_total = 0 # total number of samples
        self.confid_int = array([-inf, inf])  # confidence interval for solution
        super().__init__()

    def update_data(self, integrand, measure):
        """
        Update data

        Args:
            integrand (Integrand): an instance of Integrand
            measure (Measure): an instance of Measure

        Returns:
            None
        """
        t_start = process_time()  # time integrand evaluation
        set_x = measure.gen_samples(n_min=self.n_total,n_max=self.n)
        for r in range(self.replications):
            y = integrand.f(set_x[r])
            previous_sum_y = self.muhat_r[r] * self.n_total
            self.muhat_r[r] = (y.sum() + previous_sum_y) / self.n  # updated integrand-replication mean
        self.muhat = self.muhat_r.mean()  # mean of replication streams means
        self.sighat = self.muhat_r.std()
        self.t_eval = max(process_time() - t_start, EPS)
        self.n_total = self.n  # updated the total evaluations
        # standard deviation of stream means
        self.solution = self.muhat.copy() # mean of integrand approximations
