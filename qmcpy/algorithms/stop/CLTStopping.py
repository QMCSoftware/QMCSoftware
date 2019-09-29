''' Originally developed in MATLAB by Fred Hickernell. Translated to python by Sou-Cheng T. Choi and Aleksei Sorokin '''
from time import time

from numpy import array, ceil, full, inf, maximum, minimum, tile, zeros
from scipy.stats import norm

from . import stoppingCriterion
from ..accum_data.MeanVarData import MeanVarData


class CLTStopping(stoppingCriterion):
    ''' Stopping criterion based on the Centeral Limit Theorem. '''
    def __init__(self,distribObj,inflate=1.2,alpha=0.01,absTol=None,relTol=None,nInit=None,nMax=None):
        discDistAllowed = ["IIDDistribution"] # which discrete distributions are supported
        super().__init__(distribObj,discDistAllowed,absTol=absTol,relTol=relTol,nInit=nInit,nMax=nMax)
        self.inflate = inflate  # inflation factor
        self.alpha = alpha # uncertainty level
        self.nLevels = len(distribObj)
        self.dataObj = self.dataObj=MeanVarData(self.nLevels)

    def stopYet(self,funObj):
        if self.dataObj.stage == 'begin':  # initialize
            self.dataObj._timeStart = time()  # keep track of time
            nf = len(funObj)
            self.dataObj.prevN = zeros(nf)  # initialize data object
            self.dataObj.nextN = tile(self.nInit,nf)
            self.dataObj.muhat = full(nf,inf)
            self.dataObj.sighat = full(nf,inf)
            self.dataObj.nSigma = self.nInit  # use initial samples to estimate standard deviation
            self.dataObj.costF = zeros(nf)
            self.dataObj.stage = 'sigma'  # compute standard deviation next
        elif self.dataObj.stage == 'sigma':
            self.dataObj.prevN = self.dataObj.nextN  # update place in the sequence
            tempA = (self.dataObj.costF)**.5  # use cost of function values to decide how to allocate
            tempB = (tempA * self.dataObj.sighat).sum(0)  # samples for computation of the mean            
            nM = ceil(tempB*(-norm.ppf(self.alpha/2)*self.inflate / max(self.absTol, self.dataObj.solution*self.relTol))**2
                * (self.dataObj.sighat/self.dataObj.costF**.5))
            self.dataObj.nMu = minimum(maximum(self.dataObj.nextN,nM),self.nMax-self.dataObj.prevN)
            self.dataObj.nextN = self.dataObj.nMu + self.dataObj.prevN
            self.dataObj.stage = 'mu'  # compute sample mean next
        elif self.dataObj.stage == 'mu':
            self.dataObj.solution = self.dataObj.muhat.sum(0)
            self.dataObj.nSamplesUsed = self.dataObj.nextN
            errBar = -norm.ppf(self.alpha/2) * self.inflate * (self.dataObj.sighat**2 / self.dataObj.nMu).sum(0)**.5
            self.dataObj.confidInt = self.dataObj.solution + errBar * array([-1, 1])
            self.dataObj.stage = 'done'  # finished with computation
        self.dataObj.timeUsed = time() - self.dataObj._timeStart
        return
