""" Definition for MeanVarDataRep, a concrete implementation of AccumData """

from time import process_time
from numpy import finfo, float32, zeros
from . import AccumData

EPS = finfo(float32).eps


class MeanVarDataRep(AccumData):
    """Accumulated data Repeated Central Limit Stopping Criterion (CLTRep) \
        calculations.
    """

    def __init__(self, n_integrands, replications):
        """
        Initialize data instance

        Args:
            n_integrands (int): number of integrands
            replications (int): number of random nxm matrices to generate
        """
        super().__init__()
        self.n_integrands = n_integrands
        self.r = replications  # Number of random nxm matrices to generate
        self.muhat = zeros(self.n_integrands)
        # mean of replications means for each integrand
        self.sighat = zeros(self.n_integrands)
        # standard deviation of replications means for each integrand
        self.t_eval = zeros(self.n_integrands)
        # time used to evaluate each integrand
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
        muhat_r = zeros(self.r)  # sample mean of each nxm matrix
        for i in range(len(true_measure)):
            if self.n[i] == 0:
                continue  # integrand already sufficiently approximated
            t_start = process_time()  # time integrand evaluation
            set_x = true_measure[i].gen_tm_samples(self.r, self.n[i])
            for r in range(self.r):
                y = integrand[i].f(set_x[r])
                # Evaluate transformed function
                muhat_r[r] = y.mean()  # stream mean
            self.t_eval[i] = max(process_time() - t_start, EPS)
            self.muhat[i] = muhat_r.mean()  # mean of stream means
            self.sighat[i] = muhat_r.std()
        self.n_total += self.n.sum()
            # standard deviation of stream means
        self.solution = self.muhat.sum()  # mean of integrand approximations
