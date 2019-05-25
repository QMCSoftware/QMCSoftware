from time import time
from numpy import zeros, full, inf, sqrt, ones, kron, divide, square, array, matmul,maximum,minimum
from math import ceil
from scipy.stats import norm
import sys

from stoppingCriterion import stoppingCriterion
from meanVarData import meanVarData

class CLTStopping(stoppingCriterion):
    ''' Stopping criterion based on the Centeral Limit Theorem. '''
    def __init__(self):
        super().__init__()
        self.discDistAllowed = ["IIDDistribution"] # which discrete distributions are supported
        self.decompTypeAllowed = ["single", "multi"] # which decomposition types are supported
        self.inflate = 1.2  # inflation factor
        self.alpha = 0.01

    def stopYet(self, dataObj=[meanVarData()], funObj=[], distribObj=[]):
        if dataObj.stage == 'begin':  # initialize
            dataObj.timeStart = time()  # keep track of time
            if not type(distribObj).__name__ in self.discDistAllowed:
                raise Exception('Stopping criterion not compatible with sampling distribution')
            nf=len(funObj)
            distribObj.initStreams(nf)  # need an IIDDistribution stream for each function
            dataObj.prevN = zeros(nf)  # initialize data object
            dataObj.nextN = kron(ones((1, nf)), self.nInit)  # repmat(self.nInit, 1, nf)
            if dataObj.nextN.shape == (1, 1): dataObj.nextN = dataObj.nextN[0,]
            dataObj.muhat = full(nf,inf)
            dataObj.sighat = full(nf,inf)
            dataObj.nSigma = self.nInit  # use initial samples to estimate standard deviation
            dataObj.costF = zeros(nf)
            dataObj.stage = 'sigma'  # compute standard deviation next
        elif dataObj.stage == 'sigma':
            dataObj.prevN = dataObj.nextN  # update place in the sequence
            tempA = (dataObj.costF)**.5  # use cost of function values to decide how to allocate
            tempB = (tempA * dataObj.sighat).sum(0)  # samples for computation of the mean
            gentol = max(self.absTol, dataObj.solution * self.relTol)
            if (gentol > 0) and (dataObj.costF > 0):
              nM = ceil((tempB * ((self.getQuantile() * self.inflate / gentol)** 2)) * divide(dataObj.sighat, sqrt(dataObj.costF)))
            else:
              nM = sys.maxsize
            dataObj.nMu = minimum(maximum(dataObj.nextN, nM), self.nMax - dataObj.prevN)
            dataObj.nextN = dataObj.nMu + dataObj.prevN
            dataObj.stage = 'mu'  # compute sample mean next
        elif dataObj.stage == 'mu':
            dataObj.solution = dataObj.muhat.sum(0)
            dataObj.nSamplesUsed = dataObj.nextN
            errBar = self.get_quantile() * self.inflate * sqrt(sum(dataObj.sighat**2 / dataObj.nMu))
            dataObj.confidInt = dataObj.solution + errBar * array([-1, 1])
            dataObj.stage = 'done'  # finished with computation
        dataObj.timeUsed = time() - dataObj.timeStart
        return dataObj, distribObj

    def get_quantile(self):
        return -norm.ppf(self.alpha / 2)


if __name__ == "__main__":
    # Run Doctests
    import doctest
    x = doctest.testfile("Tests/dt_CLTStopping.py")
    print("\n" + str(x))
