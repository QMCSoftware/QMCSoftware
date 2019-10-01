from time import process_time
from numpy import arange, finfo, float32, std, zeros

from algorithms.distribution import DiscreteDistribution
from algorithms.integrand import Integrand
from . import AccumData

eps = finfo(float32).eps

class MeanVarData(AccumData):
    """
    Accumulated data for IIDDistribution calculations,
    stores the sample mean and variance of integrand values
    """
    
    def __init__(self, num_integrands):
        """ num_integrands = number of  integrands """
        super().__init__()
        self.muhat = zeros(num_integrands) # sample mean
        self.sighat = zeros(num_integrands) # sample standard deviation
        self.nSigma = zeros(num_integrands) # number of samples used to compute the sample standard deviation
        self.nMu = zeros(num_integrands)  # number of samples used to compute the sample mean

    def update_data(self, distrib_obj: DiscreteDistribution, fun_obj: Integrand):
        """
        Update data

        Args:
            distrib_obj: an instance of DiscreteDistribution
            fun_obj: an instance of Integrand

        Returns:
            None
        """
        for ii in range(len(fun_obj)):
            tStart = process_time()  # time the integrand values
            dim = distrib_obj[ii].trueD.dimension
            distrib_data = distrib_obj[ii].gen_distrib(self.nextN[ii], dim)
            y = fun_obj[ii].f(distrib_data, arange(1, dim + 1))
            self.cost_eval[ii] = max(process_time()-tStart,eps)  # to be used for multi-level methods
            if self.stage == 'sigma':
                self.sighat[ii] = std(y)  # compute the sample standard deviation if required
            self.muhat[ii] = y.mean(0)  # compute the sample mean
            self.solution = self.muhat.sum(0) # which also acts as our tentative solution
        return