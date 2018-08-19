from time import time
from numpy import zeros, full, inf, sqrt, ones, kron, divide, square, array
from math import ceil
from scipy.stats import norm
import sys

from Stopping_Criterion import Stopping_Criterion as Stopping_Criterion
from meanVar import meanVar as meanVar

class CLT(Stopping_Criterion):

    def __init__(self):
        # self.discDistAllowed = "IIDDistribution"
        # self.decompTypeAllowed = ["single", "multi"]
        self.inflate = 1.2  # inflation factor
        self.alpha = 0.01
        super().__init__()

    @property
    def discDistAllowed(self):
        return "IID"

    @property
    def decompTypeAllowed(self):  # which discrete distributions are supported
        return ["single", "multi"]

    def stopYet(self, dataObj=[], funObj=[], distribObj=[]):
        # defaults dataObj to meanVarData if not supplied by user
        if (not isinstance(dataObj, meanVar)) or (dataObj == []):
            dataObj = meanVar()

        if dataObj.stage == 'begin':  # initialize
            dataObj.timeStart = time()  # keep track of time
            if not distribObj.__class__.__name__ in [self.discDistAllowed]:
                raise Exception('Stoppoing criterion not compatible with sampling distribution')
            nf = 1
            if type(funObj) == list:
                nf = len(funObj)  # number of functions whose integrals add up to the solution # NOT SURE HOW THIS WORKS!!!
            else:
                funObj = [funObj]
            distribObj.initStreams(nf)  # need an IID stream for each function
            dataObj.prevN = zeros(nf)  # initialize data object
            # if dataObj.prevN.shape == (1,): dataObj.prevN = dataObj.prevN[0]
            dataObj.nextN = kron(ones((1, nf)), self.nInit)  # repmat(self.nInit, 1, nf)
            if dataObj.nextN.shape == (1, 1): dataObj.nextN = dataObj.nextN[0,]
            dataObj.muhat = full((1, nf), inf)
            dataObj.sighat = full((1, nf), inf)
            dataObj.nSigma = self.nInit  # use initial samples to estimate standard deviation
            dataObj.costF = zeros(nf)
            dataObj.stage = 'sigma'  # compute standard deviation next
        elif dataObj.stage == 'sigma':
            dataObj.prevN = dataObj.nextN  # update place in the sequence
            tempA = sqrt(dataObj.costF)  # use cost of function values to decide how to allocate
            tempB = sum(tempA * dataObj.sighat)  # samples for computation of the mean
            gentol = max(self.absTol, dataObj.solution * self.relTol)
            if (gentol > 0) and (dataObj.costF > 0):
              nM = ceil((tempB * ((self.getQuantile() * self.inflate / gentol)** 2)) * divide(dataObj.sighat, sqrt(dataObj.costF)))
            else:
              nM = sys.maxsize
            dataObj.nMu = min(max(dataObj.nextN, nM), self.nMax - dataObj.prevN)
            dataObj.nextN = dataObj.nMu + dataObj.prevN
            dataObj.stage = 'mu'  # compute sample mean next
        elif dataObj.stage == 'mu':
            dataObj.solution = float(sum(dataObj.muhat))
            dataObj.nSamplesUsed = dataObj.nextN
            errBar = self.getQuantile() * self.inflate * sqrt(sum(square(dataObj.sighat) / dataObj.nMu))
            dataObj.errorBound = dataObj.solution + errBar * array([-1, 1])
            dataObj.stage = 'done'  # finished with computation
        dataObj.timeUsed = time() - dataObj.timeStart
        return self, dataObj, distribObj

    def getQuantile(self):
        value = -norm.ppf(self.alpha / 2)
        return value


if __name__ == "__main__":
    # Run Doctests
    import doctest
    x = doctest.testfile("dt_CLT.py")
    print("\n" + str(x))
