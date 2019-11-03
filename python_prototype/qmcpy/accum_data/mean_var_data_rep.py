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
        self.muhat_ir = zeros((self.n_integrands,self.r)) 
        # store sample mean of ith integrand at rth replication
        self.muhat = zeros(self.n_integrands)
        # mean of replications means for each integrand
        self.sighat = zeros(self.n_integrands)
        # standard deviation of replications means for each integrand
        self.t_eval = zeros(self.n_integrands)
        # time used to evaluate each integrand
        self.n_total = zeros(self.n_integrands)

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
            t_start = process_time()  # time integrand evaluation
            n_gen = self.n[i] - self.n_total[i]
            set_x = true_measure[i].gen_tm_samples(self.r, n_gen)
            for r in range(self.r):
                y = integrand[i].f(set_x[r])
                previous_sum_y = self.muhat_ir[i,r]*self.n_total[i] # previous mean times 
                # Evaluate transformed function
                self.muhat_ir[i,r] = (y.sum()+previous_sum_y)/self.n[i]  # updated integrand-replication mean
            self.muhat[i] = self.muhat_ir[i,:].mean()  # mean of stream means
            self.sighat[i] = self.muhat_ir[i,:].std()
            self.t_eval[i] = max(process_time() - t_start, EPS)
        self.n_total = self.n.copy() # updated the total evaluations
            # standard deviation of stream means
        self.solution = self.muhat.sum()  # mean of integrand approximations
