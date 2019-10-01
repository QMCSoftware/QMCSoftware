from time import process_time

from algorithms.distribution import DiscreteDistribution
from algorithms.function import Fun
from numpy import arange, finfo, float32, std, zeros

from . import AccumData

eps = finfo(float32).eps

class MeanVarData(AccumData):
    '''
    Accumulated data for IIDDistribution calculations,
    stores the sample mean and variance of function values
    '''
    
    def __init__(self, nf: int) -> None:
        ''' nf = # function '''
        super().__init__()
        self.muhat = zeros(nf) # sample mean
        self.sighat = zeros(nf) # sample standard deviation
        self.nSigma = zeros(nf) # number of samples used to compute the sample standard deviation
        self.nMu = zeros(nf)  # number of samples used to compute the sample mean

    def update_data(self, distrib_obj: DiscreteDistribution, fun_obj: Fun) -> None:
        """
        Update data

        Args:
            distrib_obj: an instance of DiscreteDistribution
            fun_obj: an instance of function

        Returns:
            None
        """
        for ii in range(len(fun_obj)):
            tStart = process_time()  # time the function values
            dim = distrib_obj[ii].trueD.dimension
            distribData = distrib_obj[ii].gen_distrib(self.nextN[ii], dim)
            y = fun_obj[ii].f(distribData, arange(1, dim + 1))
            self.costF[ii] = max(process_time()-tStart,eps)  # to be used for multi-level methods
            if self.stage == 'sigma':
                self.sighat[ii] = std(y)  # compute the sample standard deviation if required
            self.muhat[ii] = y.mean(0)  # compute the sample mean
            self.solution = self.muhat.sum(0) # which also acts as our tentative solution
        return
