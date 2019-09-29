''' Originally developed in MATLAB by Fred Hickernell. Translated to python
by Sou-Cheng T. Choi and Aleksei Sorokin '''
from time import process_time

from algorithms.distribution import DiscreteDistribution
from algorithms.function import Fun
from numpy import arange, finfo, float32, ones, zeros

from . import AccumData

eps = finfo(float32).eps


class MeanVarDataRep(AccumData):
    ''' Accumulated data for lattice calculations '''

    def __init__(self, nf, J):
        '''
        nf = # function
        J = # streams
        '''
        super().__init__()
        self.J = J
        self.muhat = zeros(self.J)
        self.mu2hat = zeros(nf)
        self.sig2hat = zeros(nf)
        self.flags = ones(nf)

    def update_data(self, distrib_obj: DiscreteDistribution, fun_obj: Fun) -> None:
        """
        Update data

        Args:
            distrib_obj: an instance of DiscreteDistribution
            fun_obj: an instance of function

        Returns:
            None

        """
        for i in range(len(fun_obj)):
            if self.flags[
                i] == 0:  # mean of fun_obj[i] already sufficiently estimated
                continue
            tStart = process_time()  # time the function values
            dim = distrib_obj[i].trueD.dimension
            set_x = distrib_obj[i].gen_distrib(self.nextN[i], dim,
                                               self.J)  # set of j
            # distribData_{nxm}
            for j in range(self.J):
                y = fun_obj[i].f(set_x[j], arange(1, dim + 1))
                self.muhat[j] = y.mean(0)
            self.costF[i] = max(process_time() - tStart, eps)
            self.mu2hat[i] = self.muhat.mean(0)
            self.sig2hat[i] = self.muhat.std(0)
        self.solution = self.mu2hat.sum(0)
        return self
