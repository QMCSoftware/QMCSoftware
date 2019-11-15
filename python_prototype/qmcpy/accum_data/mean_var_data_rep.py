""" Definition for MeanVarDataRep, a concrete implementation of AccumData """

from ._accum_data import AccumData

from time import process_time
from numpy import zeros, nan, full, tile, array, finfo, float32, inf

EPS = finfo(float32).eps


class MeanVarDataRep(AccumData):
    """Accumulated data Repeated Central Limit Stopping Criterion (CLTRep) \
        calculations.
    """

    def __init__(self, levels, n_init, replications):
        """
        Initialize data instance

        Args:
            levels (int): number of integrands
            n_init (int): initial number of samples
            replications (int): number of random nxm matrices to generate
        """
        self.r = replications  # Number of random nxm matrices to generate
        self.muhat_ir = zeros((levels, self.r)) 
        self.solution = nan
        self.muhat = full(levels, inf) # sample mean
        self.sighat = full(levels, inf) # sample standard deviation
        self.t_eval = zeros(levels) # processing time for each integrand
        self.n = tile(n_init, levels).astype(float) # currnet number of samples
        self.n_total = tile(0, levels) # total number of samples
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
    
    def __repr__(self):
        """
        Print important attribute values

        Args: 
            attributes (list): list of attributes to print
        
        Returns:
            string of self info
        """
        return super().__repr__(['r'])