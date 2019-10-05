"""
Definition for MeanVarDataRep, subclass of AccumData
"""
from time import process_time

from qmcpy.distribution import DiscreteDistribution
from qmcpy.integrand import Integrand
from numpy import arange, finfo, float32, ones, zeros

from . import AccumData

EPS = finfo(float32).eps

class MeanVarDataRep(AccumData):
    """ Accumulated data for lattice calculations """

    def __init__(self, n_integrands, n_streams):
        """
        Initialize data instance

        Args:
            n_integrands (int): number of integrands
            n_streams (int): number of random nxm matricies to generate

        """
        super().__init__()
        self.n_integrands = n_integrands
        self.n_streams = n_streams # Number of random nxm matricies to generate
        self.muhat = zeros(self.n_streams) # sample mean of each nxm matrix
        self.mu2hat = zeros(self.n_integrands) # mean of n_streams means for each integrand
        self.sig2hat = zeros(self.n_integrands) # standard deviation of n_streams means for each integrand
        self.flag = ones(self.n_integrands) # flag when an integrand has been sufficiently approximated
        self.t_eval = zeros(self.n_integrands) # time used to evaluate each integrand

    def update_data(self, distribution, integrand):
        """
        Update data

        Args:
            distribution (DiscreteDistribution): an instance of DiscreteDistribution
            integrand (Integrand): an instance of Integrand

        Returns:
            None

        """
        for i in range(self.n_integrands):
            if self.flag[i] == 0:
                continue # integrand already sufficiently approximated
            t_start = process_time()  # time integrand evaluation
            dim = distribution[i].true_distribution.dimension # dimension of the integrand
            # set_x := n_streams matricies housing nxm integrand values
            set_x = distribution[i].gen_distrib(self.n_next[i], dim, self.n_streams)
            for j in range(self.n_streams):
                y = integrand[i].f(set_x[j], arange(1, dim + 1)) # Evaluate transformed function
                self.muhat[j] = y.mean() # stream mean
            self.t_eval[i] = max(process_time()-t_start, EPS)
            self.mu2hat[i] = self.muhat.mean() # mean of stream means
            self.sig2hat[i] = self.muhat.std() # standard deviation of stream means
        self.solution = self.mu2hat.sum() # mean of integrand approximations