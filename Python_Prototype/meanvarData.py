from accumData import accumData
import numpy as np
from time import time


# Accumulated data for IID calculations, stores the sample mean and
# variance of function values
class meanVarData(accumData):


    def __init__(self):
        """
        >>> mvd = meanVarData()
        >>> print(mvd.__dict__)
        {'muhat': [], 'sighat': [], 'nSigma': [], 'nMu': [], 'solution': nan, 'stage': 'begin', 'prevN': [], 'nextN': [], 'timeUsed': [], 'nSamplesUsed': [], 'errorBound': [-inf, inf], 'costF': []}
        """
        self.muhat = [] # sample mean
        self.sighat = [] # sample standard deviation
        self.nSigma = [] # number of samples used to compute the sample standard deviation
        self.nMu = []  # number of samples used to compute the sample mean
        super().__init__()


    def timeStart(self):  # starting time
        """
        >>> mvd = meanVarData()
        >>> mvd.__timeStart # doctest:+ELLIPSIS
        Traceback (most recent call last):
          ...
        AttributeError: 'meanVarData' object has no attribute '__timeStart'
        >>> mvd.timeStart()  # doctest:+ELLIPSIS
        >>> mvd._meanVarData__timeStart  # doctest:+ELLIPSIS
        1...
        """
        self.__timeStart = time()
        #print(self.__timeStart) # "hidden" property, but it can still be exposed somehow as the last doctest shows
        return


    def updateData(self, distribObj, funObj):
        nf = 1
        if type(funObj) == list:
            nf = len(funObj)
        else:
            funObj = [funObj]
        # preallocate vectors
        self.solution = np.zeros(nf)
        self.sighat = np.zeros(nf)
        self.costF = np.zeros(nf)
        for ii in range(0, nf):
            tStart = time()  # time the function values
            y = funObj[ii].f(distribObj.genDistrib(self.prevN[ii] + 1, self.prevN[ii] + self.nextN[ii], self.nextN[ii],
                                               range(0, funObj[ii].dimension), ii),
                         range(0, funObj[ii].dimension))
            self.costF[ii] = time() - tStart  # to be used for multi-level methods
            if self.stage == 'sigma':
                self.sighat[ii] = np.std(y)  # compute the sample standard deviation if required
            self.muhat[ii] = np.mean(y)  # compute the sample mean
            self.solution = sum(self.muhat)  # which also acts as our tentative solution

        return self

if __name__ == "__main__":
    import doctest
    doctest.testmod()