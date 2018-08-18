from time import time
from numpy import zeros, full, inf, sqrt, ones, kron, divide, square
from math import ceil
from scipy.stats import norm

from Stopping_Criterion import Stopping_Criterion


class CLT(Stopping_Criterion):
    
    def __init__(self):
        #self.discDistAllowed = "IIDDistribution"  
        #self.decompTypeAllowed = ["single", "multi"]
        self.inflate = 1.2  # inflation factor
        self.alpha = 0.01
        super().__init__()
    
    @property
    def discDistAllowed(self): 
        return "IIDDistribution"

    @property
    def decompTypeAllowed(self): # which discrete distributions are supported
        return ["single", "multi"]

    def stopYet(self,dataObj,distribObj,funObj):
        # defaults dataObj to meanVarData if not supplied by user
        if dataObj.stage == 'begin':  # initialize
            dataObj.timeStart = time()  # keep track of time
            if distribObj.__class__.__name__ not in self.discDistAllowed:
                raise Exception('Stoppoing criterion not compatible with sampling distribution')
            nf = 1
            if type(funObj) == list:
                nf = len(funObj)  # number of functions whose integrals add up to the solution # NOT SURE HOW THIS WORKS!!!
            else:
                funObj = [funObj]
            distribObj.initStreams(nf)  # need an IID stream for each function
            dataObj.prevN = zeros(nf)  # initialize data object
            dataObj.nextN = kron(ones((1, nf)), self.nInit)  # repmat(self.nInit, 1, nf)
            dataObj.muhat = full((1, nf), inf)
            dataObj.sighat = full((1, nf), inf)
            dataObj.nSigma = self.nInit  # use initial samples to estimate standard deviation
            dataObj.costF = zeros(nf)
            dataObj.stage = 'sigma'  # compute standard deviation next
        elif dataObj.stage == 'sigma':
            dataObj.prevN = dataObj.nextN  # update place in the sequence
            tempA = sqrt(dataObj.costF)  # use cost of function values to decide how to allocate
            tempB = sum(tempA * dataObj.sighat)  # samples for computation of the mean            
            
            nM = ceil((tempB * ((self.getQuantile() * self.inflate / max(self.absTol, dataObj.solution * self.relTol))** 2))
            * divide(dataObj.sighat, sqrt(dataObj.costF)))

            dataObj.nMu = min(max(dataObj.nextN, nM), self.nMax - dataObj.prevN)
            dataObj.nextN = dataObj.nMu + dataObj.prevN
            dataObj.stage = 'mu'  # compute sample mean next
        elif dataObj.stage == 'mu':
            dataObj.solution = sum(dataObj.muhat)
            dataObj.nSamplesUsed = dataObj.nextN
            errBar = (self.getQuantile() * self.inflate) * sqrt(sum(square(dataObj.sighat) / dataObj.nMu))
            dataObj.errorBound = dataObj.solution + errBar * [-1, 1]
            dataObj.stage = 'done'  # finished with computation
        dataObj.timeUsed = time() - dataObj.timeStart
        return dataObj, distribObj

    def getQuantile(self):
        value = -norm.pdf(self.alpha / 2)
        return value

if __name__ == "__main__":
    # Run Doctests
    import doctest
    x = doctest.testfile("dt_CLTStopping.py")
    print("\n"+str(x))
