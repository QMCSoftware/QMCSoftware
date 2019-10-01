from time import time
from numpy import array, full, zeros
from scipy.stats import norm

from algorithms.distribution import DiscreteDistribution
from algorithms.integrand import Integrand
from . import StoppingCriterion
from ..accum_data.MeanVarDataRep import MeanVarDataRep
from .. import MaxSamplesWarning

class CLTRep(StoppingCriterion):
    """ Stopping criterion based on var(stream_1_estimate,...,stream_16_estimate)<errorTol """
    def __init__(self, distrib_obj: DiscreteDistribution, n_streams=16, inflate=1.2, alpha=0.01, abs_tol=1e-2, rel_tol=0, n_init=1024, n_max=1e8):
        """
        Args:
            distrib_obj: an instance of DiscreteDistribution
            n_streams: number of random nxm matricies to generate
            inflate: inflation factor when estimating variance
            alpha: significance level for confidence interval
            abs_tol: absolute error tolerance
            rel_tol: relative error tolerance
            n_init: initial number of samples
            n_max: maximum number of samples
        """
        allowed_distribs = ["QuasiRandom"] # supported distributions
        super().__init__(distrib_obj, allowed_distribs, abs_tol, rel_tol, n_init, n_max)
        self.inflate = inflate  # inflation factor
        self.alpha = alpha # uncertainty level
        self.stage = 'begin'
        # Construct Data Object
        n_integrands = len(distrib_obj)
        self.data_obj = MeanVarDataRep(n_integrands, n_streams) # house integration data
        self.data_obj.n_prev = zeros(n_integrands) # previous n used for each integrand
        self.data_obj.n_next = full(n_integrands,self.n_init) # next n to be used for each integrand
    
    def stopYet(self,funObj: Integrand):
        """
        Determine when to stop

        Args:
            distrib_obj: an instance of DiscreteDistribution

        Returns:
            None

        """
        for i in range(self.data_obj.n_integrands):
            if self.data_obj.sig2hat[i] < self.abs_tol: # Sufficient estimate for mean of funObj[i]
                self.data_obj.flag[i] = 0 # Stop estimation of i_th integrand
            else: # Double n for next sample
                self.data_obj.n_prev[i] = self.data_obj.n_next[i]
                self.data_obj.n_next[i] = self.data_obj.n_prev[i]*2
        if self.data_obj.flag.sum()==0 or self.data_obj.n_next.max() > self.n_max:
            # Stopping Criterion Met
            if self.data_obj.n_next.max() > self.n_max: 
                raise MaxSamplesWarning("Max number of samples used. Tolerance may not be met")
            self.data_obj.n_samples_total = self.data_obj.n_next
            errBar = -norm.ppf(self.alpha / 2) * self.inflate * (self.data_obj.sig2hat**2 / self.data_obj.n_next).sum(0)**.5
            self.data_obj.confidInt = self.data_obj.solution + errBar * array([-1, 1]) # CLT confidence interval
            self.stage = 'done'  # finished with computation
        return