import accumData
import numpy as np
from time import time


# Accumulated data for IID calculations, stores the sample mean and
# variance of function values
class meanVarData(accumData):

    def __init__(self):
        super(accumData, self).__init__()
        self.muhat  # sample mean
        self.sighat  # sample standard deviation
        self.nSigma  # number of samples used to compute the sample standard deviation
        self.nMu  # number of samples used to compute the sample mean

    def updateData(self, distribObj, funObj):
        nf = len(funObj)
        # preallocate vectors
        self.solution = np.zeros(nf)
        self.sighat = np.zeros(nf)
        self.costF = np.zeros(nf)
        for ii in range(0, nf):
            tStart = time()  # time the function values
            y = funObj.f(funObj[ii],
                         distribObj.genDistrib(self.prevN[ii] + 1, self.prevN[ii] + self.nextN[ii], self.nextN[ii],
                                               range(0, funObj[ii].dimension), ii),
                         range(0, funObj[ii].dimension))
            self.costF[ii] = time() - tStart  # to be used for multi-level methods
            if self.stage == 'sigma':
                self.sighat[ii] = np.std(y)  # compute the sample standard deviation if required
            self.muhat[ii] = np.mean(y)  # compute the sample mean
            self.solution = sum(self.muhat)  # which also acts as our tentative solution

        return self
