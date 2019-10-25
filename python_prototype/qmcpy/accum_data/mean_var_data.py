""" Definition of MeanVarData, a concrete implementation of AccumData """

from time import process_time
from numpy import finfo, float32, full, inf, std, zeros
from . import AccumData

EPS = finfo(float32).eps


class MeanVarData(AccumData):
    """
    Accumulated data for IIDDistribution calculations,
    and store the sample mean and variance of integrand values
    """

    def __init__(self, n_integrands):
        """
        Initialize data instance

        Args:
            n_integrands (int): number of integrands
        """
        super().__init__()
        self.n_integrands = n_integrands
        self.muhat = full(self.n_integrands, inf)  # sample mean
        self.sighat = full(self.n_integrands, inf)  # sample standard deviation
        self.t_eval = zeros(self.n_integrands)
        # time used to evaluate each integrand

    def update_data(self, integrand, true_measure):
        """
        Update data

        Args:
            true_measure (TrueMeasure): an instance of TrueMeasure
            integrand (Integrand): an instance of Integrand

        Returns:
            None
        """
        for i, (integrand_i, true_measure_i) in enumerate(zip(integrand, true_measure)):
            t_start = process_time()  # time the integrand values
            set_x = true_measure_i.gen_tm_samples(1, self.n_next[i]).squeeze(0)
            y = integrand_i.f(set_x)
            self.t_eval[i] = max(process_time() - t_start, EPS)
            # for multi-level methods
            self.sighat[i] = std(y)  # compute the sample standard deviation
            self.muhat[i] = y.mean(0)  # compute the sample mean
            self.solution = self.muhat.sum()
            # which also acts as our tentative solution
