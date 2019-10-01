from time import process_time
from numpy import arange, finfo, float32, ones, zeros

from algorithms.distribution import DiscreteDistribution
from algorithms.integrand import Integrand
from . import AccumData

eps = finfo(float32).eps

class MeanVarDataRep(AccumData):
    """ Accumulated data for lattice calculations """

    def __init__(self, num_integrands, n_streams):
        """
        num_integrands = number of integrands
        n_streams = number of streams
        """
        super().__init__()
        self.n_streams = n_streams # Number of random nxm matricies to generate
        self.muhat = zeros(self.n_streams) # sample mean of each nxm matrix
        self.mu2hat = zeros(num_integrands) # mean of n_streams means for each integrand
        self.sig2hat = zeros(num_integrands) # standard deviation of n_streams means for each integrand
        self.flag = ones(num_integrands) # flag when an integrand has been sufficiently approximated 

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
            if self.flag[i] == 0: continue # integrand already sufficiently approximated
            t_start = process_time()  # time integrand evaluation
            dim = distrib_obj[i].trueD.dimension # dimension of the integrand
            # set_x := n_streams matricies housing nxm integrand values
            set_x = distrib_obj[i].gen_distrib(self.nextN[i], dim, self.n_streams)  
            for j in range(self.n_streams):
                y = fun_obj[i].f(set_x[j], arange(1, dim+1)) # Evaluate transformed function
                self.muhat[j] = y.mean() # stream mean
            self.cost_eval[i] = max(process_time()-t_start, eps)
            self.mu2hat[i] = self.muhat.mean() # mean of stream means
            self.sig2hat[i] = self.muhat.std() # standard deviation of stream means
        self.solution = self.mu2hat.sum() # mean of integrand approximations
        return self