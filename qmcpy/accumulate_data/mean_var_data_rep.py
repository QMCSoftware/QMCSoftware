from ._accumulate_data import AccumulateData
from numpy import *


class MeanVarDataRep(AccumulateData):
    """
    Update and store mean and variance estimates with repliations. 
    See the stopping criterion that utilize this object for references.
    """

    parameters = ['replications','solution','sighat','n_total','error_bound','confid_int']

    def __init__(self, stopping_criterion, integrand, n_init, replications):
        """
        Args:
            stopping_criterion (StoppingCriterion): a StoppingCriterion instance
            integrand (Integrand): an Integrand instance    
            n_init (int): initial number of samples
            replications (int): number of replications
        """
        # Extract QMCPy objects
        self.stopping_criterion = stopping_criterion
        self.integrand = integrand
        self.measure = self.integrand.measure
        self.distribution = self.measure.distribution
        # Set Attributes
        self.replications = replications
        self.muhat_r = zeros(int(self.replications))
        self.solution = nan
        self.muhat = inf  # sample mean
        self.sighat = inf # sample standard deviation
        self.t_eval = 0  # processing time for each integrand
        self.n_r = n_init  # current number of samples to draw from discrete distribution
        self.n_r_prev = 0 # previous number of samples drawn from discrete distributoin
        self.n_total = 0 # total number of samples across all replications
        self.confid_int = array([-inf, inf])  # confidence interval for solution
        # get seeds for each replication
        seed = self.distribution.seed
        if isinstance(seed,int) or isinstance(seed, uint64):
            random.seed(seed)
        else:
            random.seed(seed[0])
        self.seeds = random.randint(0,100000,int(self.replications))
        super(MeanVarDataRep,self).__init__()

    def update_data(self):
        """ See abstract method. """
        for r in range(int(self.replications)):
            self.distribution.set_seed(int(self.seeds[r]))
            x = self.distribution.gen_samples(n_min=self.n_r_prev,n_max=self.n_r)
            y = self.integrand.f(x).squeeze()
            previous_sum_y = self.muhat_r[r] * self.n_r_prev
            self.muhat_r[r] = (y.sum() + previous_sum_y) / self.n_r  # updated integrand-replication mean
        self.solution = self.muhat_r.mean()  # mean of replication means
        self.sighat = self.muhat_r.std()
        self.n_r_prev = self.n_r  # updated the total evaluations
        self.n_total = self.n_r * self.replications
