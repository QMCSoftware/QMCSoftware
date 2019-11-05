""" Definition of MeanVarData, a concrete implementation of AccumData """

from time import process_time
from numpy import finfo, float32, full, inf, zeros

from ._accum_data import AccumData

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
        self.t_eval = zeros(self.n_integrands)  # processing time for each integrand
        self.n_total = 0

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
