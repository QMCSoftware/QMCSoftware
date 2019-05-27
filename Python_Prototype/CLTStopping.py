from time import time
from numpy import zeros, full, inf, sqrt, ones, kron, divide, square, array, matmul,maximum,minimum,tile
from math import ceil
from scipy.stats import norm
import sys

from stoppingCriterion import stoppingCriterion
from meanVarData import meanVarData

class CLTStopping(stoppingCriterion):
    ''' Stopping criterion based on the Centeral Limit Theorem. '''
    def __init__(self):
        discDistAllowed = ["IIDDistribution"]
        decompTypeAllowed = ["single", "multi"]
        super().__init__(discDistAllowed,decompTypeAllowed)
        self.inflate = 1.2  # inflation factor
        self.alpha = 0.01

    def stopYet(self, dataObj=meanVarData(), funObj=[], distribObj=[]):
        if dataObj.stage == 'begin':  # initialize
            dataObj.timeStart = time()  # keep track of time
            if not type(distribObj).__name__ in self.discDistAllowed:
                raise Exception('Stopping criterion not compatible with sampling distribution')
            nf=len(funObj)
            distribObj.initStreams()  # need an IIDDistribution stream for each function
            dataObj.prevN = zeros(nf)  # initialize data object
            dataObj.nextN = tile(self.nInit,nf)
            dataObj.muhat = full(nf,inf)
            dataObj.sighat = full(nf,inf)
            dataObj.nSigma = self.nInit  # use initial samples to estimate standard deviation
            dataObj.costF = zeros(nf)
            dataObj.stage = 'sigma'  # compute standard deviation next
        elif dataObj.stage == 'sigma':
            dataObj.prevN = dataObj.nextN  # update place in the sequence
            tempA = (dataObj.costF)**.5  # use cost of function values to decide how to allocate
            tempB = (tempA * dataObj.sighat).sum(0)  # samples for computation of the mean
            nM = ceil(tempB*(self.get_quantile()*self.inflate / max(self.absTol, dataObj.solution*self.relTol))**2
                * (dataObj.sighat/dataObj.costF**.5))
            dataObj.nMu = minimum(maximum(dataObj.nextN,nM),self.nMax-dataObj.prevN)
            dataObj.nextN = dataObj.nMu + dataObj.prevN
            dataObj.stage = 'mu'  # compute sample mean next
        elif dataObj.stage == 'mu':
            dataObj.solution = dataObj.muhat.sum(0)
            dataObj.nSamplesUsed = dataObj.nextN
            errBar = self.get_quantile() * self.inflate * (dataObj.sighat**2 / dataObj.nMu).sum(0)**.5
            dataObj.confidInt = dataObj.solution + errBar * array([-1, 1])
            dataObj.stage = 'done'  # finished with computation
        dataObj.timeUsed = time() - dataObj.timeStart
        return dataObj, distribObj
    
    def get_quantile(self):
        # dependent property
        return -norm.ppf(self.alpha / 2)

if __name__ == "__main__":
    # Run Doctests
    import doctest
    x = doctest.testfile("Tests/dt_CLTStopping.py")
    print("\n" + str(x))
