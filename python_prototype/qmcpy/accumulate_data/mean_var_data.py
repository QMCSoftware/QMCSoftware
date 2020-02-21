""" Definition of MeanVarData, a concrete implementation of AccumulateData """

from ._accumulate_data import AccumulateData
from ..integrand._integrand import Integrand
from time import process_time
from numpy import array, finfo, float32, full, inf, nan, tile, zeros

EPS = finfo(float32).eps


class MeanVarData(AccumulateData):
    """
    Accumulated data for IIDDistribution calculations,
    and store the sample mean and variance of integrand values
    """

    parameters = ['levels','solution','n','n_total','confid_int']

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
        # account for multilevel
        if isinstance(self.integrand,Integrand): 
            # single level -> make it appear multi-level
            self.integrands = [self.integrand]
            self.measures = [self.measure]
        else:
            # multi-level
            self.integrands = self.integrand 
            self.measures = self.measure
        # Set Attributes
        self.levels = len(self.integrands)
        self.solution = nan
        self.muhat = full(self.levels, inf)  # sample mean
        self.sighat = full(self.levels, inf)  # sample standard deviation
        self.t_eval = zeros(self.levels)  # processing time for each integrand
        self.n = tile(n_init, self.levels) # currnet number of samples
        self.n_total = 0  # total number of samples
        self.confid_int = array([-inf, inf])  # confidence interval for solution
        super().__init__()

    def update_data(self):
        """ Update data """
        for l in range(self.levels):
            t_start = process_time()  # time the integrand values
            samples = self.measures[l].gen_samples(n=self.n[l])
            y = self.integrands[l].f(samples).squeeze()
            self.t_eval[l] = max(process_time() - t_start, EPS)
            self.sighat[l] = y.std()  # compute the sample standard deviation
            self.muhat[l] = y.mean()  # compute the sample mean
            self.n_total += self.n[l]  # add to total samples
        self.solution = self.muhat.sum()  # tentative solution
