from ._accumulate_data import AccumulateData
from ..integrand._integrand import Integrand
from time import perf_counter
from numpy import array, finfo, float32, full, inf, nan, tile, zeros

EPS = finfo(float32).eps


class MeanVarData(AccumulateData):

    parameters = ['levels','solution','n','n_total','confid_int']

    def __init__(self, stopping_criterion, integrand, n_init):
        """
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
        self.integrand = integrand
        # Set Attributes
        if self.integrand.multilevel:
            self.levels = len(self.integrand.dimensions)
        else:
            self.levels = 1
        self.solution = nan
        self.muhat = full(self.levels, inf)  # sample mean
        self.sighat = full(self.levels, inf)  # sample standard deviation
        self.t_eval = zeros(self.levels)  # processing time for each integrand
        self.n = tile(n_init, self.levels) # currnet number of samples
        self.n_total = 0  # total number of samples
        self.confid_int = array([-inf, inf])  # confidence interval for solution
        super().__init__()

    def update_data(self):
        """ See abstract method. """
        for l in range(self.levels):
            t_start = perf_counter() # time the integrand values
            if self.integrand.multilevel:
                # reset dimension
                new_dim = self.integrand.dim_at_level(l)
                self.measure.set_dimension(new_dim)
                samples = self.distribution.gen_samples(n=self.n[l])
                y = self.integrand.f(samples,l=l).squeeze()
            else:
                samples = self.distribution.gen_samples(n=self.n[l])
                y = self.integrand.f(samples).squeeze()
            self.t_eval[l] = max(perf_counter() - t_start, EPS)
            self.sighat[l] = y.std() # compute the sample standard deviation
            self.muhat[l] = y.mean() # compute the sample mean
            self.n_total += self.n[l] # add to total samples
        self.solution = self.muhat.sum() # tentative solution
