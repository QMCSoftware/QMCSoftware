from ._accumulate_data import AccumulateData
from numpy import *
from copy import deepcopy

class MeanVarDataRep(AccumulateData):
    """
    Update and store mean and variance estimates with repliations. 
    See the stopping criterion that utilize this object for references.
    """

    parameters = ['solution','replications','sighat','n','n_total','error_bound','confid_int']

    def __init__(self, stopping_crit, integrand, true_measure, discrete_distrib, n_init, replications):
        """
        Args:
            stopping_crit (StoppingCriterion): a StoppingCriterion instance
            integrand (Integrand): an Integrand instance
            true_measure (TrueMeasure): A TrueMeasure instance
            discrete_distrib (DiscreteDistribution): a DiscreteDistribution instance  
            n_init (int): initial number of samples
            replications (int): number of replications
        """
        self.stopping_crit = stopping_crit
        self.integrand = integrand
        self.true_measure = true_measure
        self.discrete_distrib = discrete_distrib
        # Set Attributes
        self.replications = int(replications)
        self.ysums = zeros((self.replications,self.integrand.dprime),dtype=float)
        self.solution = nan
        self.muhat = inf # sample mean
        self.sighat = inf # sample standard deviation
        self.t_eval = 0  # processing time for each integrand
        self.n = n_init*ones(self.integrand.dprime,dtype=float)  # current number of samples to draw from discrete distribution
        self.nprev = zeros(self.integrand.dprime,dtype=float) # previous number of samples drawn from discrete distributoin
        self.n_total = 0 # total number of samples across all replications
        self.confid_int = array([-inf, inf])  # confidence interval for solution
        # get seeds for each replication
        ld_seeds = self.discrete_distrib.rng.choice(100000,self.replications,replace=False).astype(dtype=uint64)+1
        self.ld_streams = [deepcopy(self.discrete_distrib) for r in range(self.replications)]
        for r in range(self.replications): self.ld_streams[r].set_seed(ld_seeds[r])
        self.compute_flags = ones(self.integrand.dprime)
        super(MeanVarDataRep,self).__init__()

    def update_data(self):
        """ See abstract method. """
        nmaxidx = argmax(self.n)
        n_max = self.n[nmaxidx]
        n_min = self.nprev[nmaxidx]
        for r in range(self.replications):
            x = self.ld_streams[r].gen_samples(n_min=n_min,n_max=n_max)
            if self.integrand.dprime>1:
                y = self.integrand.f(x,compute_flags=self.compute_flags)
            else:
                y = self.integrand.f(x)
            yflagged = y*self.compute_flags
            self.ysums[r] = self.ysums[r] + yflagged.sum(0)
        ymeans = self.ysums/self.n
        self.solution = ymeans.mean(0)
        self.sighat = ymeans.std(0)
        self.n_total = (self.n * self.replications).max()
