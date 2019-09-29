''' Originally developed in MATLAB by Fred Hickernell. Translated to python
by Sou-Cheng T. Choi and Aleksei Sorokin '''
from time import process_time

from algorithms.distribution import discreteDistribution
from algorithms.function import fun
from numpy import arange, finfo, float32, ones, zeros

from . import accumData

eps = finfo(float32).eps


class meanVarData_Rep(accumData):
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

    def updateData(self, distribObj: discreteDistribution, funObj: fun) -> None:
        """
        Update data

        Args:
            distribObj: an instance of discreteDistribution
            funObj: an instance of function

        Returns:
            None

        """
        for i in range(len(funObj)):
            if self.flags[
                i] == 0:  # mean of funObj[i] already sufficiently estimated
                continue
            tStart = process_time()  # time the function values
            dim = distribObj[i].trueD.dimension
            set_x = distribObj[i].genDistrib(self.nextN[i], dim,
                                             self.J)  # set of j
            # distribData_{nxm}
            for j in range(self.J):
                y = funObj[i].f(set_x[j], arange(1, dim + 1))
                self.muhat[j] = y.mean(0)
            self.costF[i] = max(process_time() - tStart, eps)
            self.mu2hat[i] = self.muhat.mean(0)
            self.sig2hat[i] = self.muhat.std(0)
        self.solution = self.mu2hat.sum(0)
        return self
