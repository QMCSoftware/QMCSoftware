''' Originally developed in MATLAB by Fred Hickernell. Translated to python by Sou-Cheng T. Choi and Aleksei Sorokin '''
from time import time

from numpy import zeros, full, inf, array, maximum, minimum, tile, ceil
from scipy.stats import norm

from algorithms.meanVarData import meanVarData
from algorithms.stoppingCriterion import stoppingCriterion


class CLTStopping(stoppingCriterion):
    ''' Stopping criterion based on the Centeral Limit Theorem. '''
    def __init__(self,inflate=1.2,alpha=0.01,absTol=None,relTol=None,nInit=None,nMax=None):
        discDistAllowed = ["IIDDistribution"] # which discrete distributions are supported
        decompTypeAllowed = ["single", "multi"] # which decomposition types are supported
        super().__init__(discDistAllowed,decompTypeAllowed,absTol=absTol,relTol=relTol,nInit=nInit,nMax=nMax)
        self.inflate = inflate  # inflation factor
        self.alpha = alpha # uncertainty level

    def stopYet(self, dataObj=None, funObj=[], distribObj=[]):
        if dataObj==None: dataObj=meanVarData(len(funObj))
        if dataObj.stage == 'begin':  # initialize
            dataObj._timeStart = time()  # keep track of time
            if type(distribObj).__name__ not in self.discDistAllowed:
                raise Exception('Stopping criterion not compatible with sampling distribution')
            nf = len(funObj)
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
        dataObj.timeUsed = time() - dataObj._timeStart
        return dataObj, distribObj
    
    def get_quantile(self):
        # dependent property
        return -norm.ppf(self.alpha / 2)

if __name__ == "__main__":
    # Run Doctests
    import doctest
    x = doctest.testfile("Tests/dt_CLTStopping.py")
    print("\n" + str(x))
