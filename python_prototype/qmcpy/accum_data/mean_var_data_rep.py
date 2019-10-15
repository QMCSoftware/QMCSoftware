""" Definition for MeanVarDataRep, a concrete implementation of AccumData """

from time import process_time
from numpy import finfo, float32, ones, zeros
from . import AccumData

EPS = finfo(float32).eps


class MeanVarDataRep(AccumData):
    """Accumulated data Repeated Central Limit Stopping Criterion (CLTRep) \
        calculations.
    """

    def __init__(self, n_integrands, n_streams):
        """
        Initialize data instance

        Args:
            n_integrands (int): number of integrands
            n_streams (int): number of random nxm matrices to generate
        """
        super().__init__()
        self.n_integrands = n_integrands
        self.n_streams = n_streams  # Number of random nxm matrices to generate
        self.muhat = zeros(self.n_streams)  # sample mean of each nxm matrix
        self.mu2hat = zeros(self.n_integrands)
        # mean of n_streams means for each integrand
        self.sig2hat = zeros(self.n_integrands)
        # standard deviation of n_streams means for each integrand
        self.flag = ones(self.n_integrands)
        # flag when an integrand has been sufficiently approximated
        self.t_eval = zeros(self.n_integrands)
        # time used to evaluate each integrand

    def update_data(self, true_measure, integrand):
        """
        Update data

        Args:
            true_measure (TrueMeasure): an instance of TrueMeasure
            integrand (Integrand): an instance of Integrand

        Returns:
            None
        """
        for i in range(self.n_integrands):
            if self.flag[i] == 0:
                continue  # integrand already sufficiently approximated
            t_start = process_time()  # time integrand evaluation
            set_x = true_measure[i].gen_true_measure_samples(
                self.n_streams, self.n_next[i])
            for j in range(self.n_streams):
                y = integrand[i].g(set_x[j])
                # Evaluate transformed function
                self.muhat[j] = y.mean()  # stream mean
            self.t_eval[i] = max(process_time() - t_start, EPS)
            self.mu2hat[i] = self.muhat.mean()  # mean of stream means
            self.sig2hat[i] = self.muhat.std()
            # standard deviation of stream means
        self.solution = self.mu2hat.sum()  # mean of integrand approximations