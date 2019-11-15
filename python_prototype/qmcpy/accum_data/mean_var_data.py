""" Definition of MeanVarData, a concrete implementation of AccumData """

from ._accum_data import AccumData

from time import *
from numpy import *

EPS = finfo(float32).eps


class MeanVarData(AccumData):
    """
    Accumulated data for IIDDistribution calculations,
    and store the sample mean and variance of integrand values
    """

    def __init__(self, levels, n_init):
        """
        Initialize data instance

        Args:
            levels (int): number of integrands
            n_init (int): initial number of samples
        """
        self.solution = nan
        self.muhat = full(levels, inf) # sample mean
        self.sighat = full(levels, inf) # sample standard deviation
        self.t_eval = zeros(levels) # processing time for each integrand
        self.n = tile(n_init, levels).astype(float) # currnet number of samples
        self.n_total = 0 # total number of samples
        self.confid_int = array([-inf, inf]) # confidence interval for solution
        super().__init__()

    def update_data(self, integrand, true_measure):
        """
        Update data

        Args:
            integrand (Integrand): an instance of Integrand
            true_measure (TrueMeasure): an instance of TrueMeasure

        Returns:
            None
        """
        for i in range(len(true_measure)):
            t_start = process_time()  # time the integrand values
            set_x = true_measure[i].gen_tm_samples(1, self.n[i]).squeeze(0)
            y = integrand[i].f(set_x).squeeze()
            self.t_eval[i] = max(process_time() - t_start, EPS)
            self.sighat[i] = y.std()  # compute the sample standard deviation
            self.muhat[i] = y.mean()  # compute the sample mean
            self.n_total += self.n[i]  # add to total samples
        self.solution = self.muhat.sum()  # tentative solution
