from time import process_time
from numpy import arange, finfo, float32, std, zeros, full, inf

from algorithms.distribution import DiscreteDistribution
from algorithms.integrand import Integrand
from . import AccumData

eps = finfo(float32).eps

class MeanVarData(AccumData):
    """
    Accumulated data for IIDDistribution calculations,
    stores the sample mean and variance of integrand values
    """
    
    def __init__(self, n_integrands):
        """ n_integrands = number of  integrands """
        super().__init__()
        self.n_integrands = n_integrands
        self.muhat = full(self.n_integrands,inf) # sample mean
        self.sighat = full(self.n_integrands,inf) # sample standard deviation
        self.t_eval = zeros(self.n_integrands) # time used to evaluate each integrand

    def update_data(self, distrib_obj: DiscreteDistribution, fun_obj: Integrand):
        """
        Update data

        Args:
            distrib_obj: an instance of DiscreteDistribution
            fun_obj: an instance of Integrand

        Returns:
            None
        """
        for i in range(len(fun_obj)):
            tStart = process_time()  # time the integrand values
            dim = distrib_obj[i].trueD.dimension
            distrib_data = distrib_obj[i].gen_distrib(self.n_next[i], dim)
            y = fun_obj[i].f(distrib_data, arange(1, dim + 1))
            self.t_eval[i] = max(process_time()-tStart,eps)  # to be used for multi-level methods
            self.sighat[i] = std(y)  # compute the sample standard deviation if required
            self.muhat[i] = y.mean(0)  # compute the sample mean
            self.solution = self.muhat.sum(0) # which also acts as our tentative solution
        return