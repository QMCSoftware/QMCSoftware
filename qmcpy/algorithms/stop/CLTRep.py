''' Originally developed in MATLAB by Fred Hickernell. Translated to python by Sou-Cheng T. Choi and Aleksei Sorokin '''
from time import time

from numpy import array, full, zeros
from scipy.stats import norm

from . import StoppingCriterion
from ..accum_data.MeanVarDataRep import MeanVarDataRep


class CLTRep(StoppingCriterion):
    ''' Stopping criterion based on var(stream_1_estimate,stream_2_estimate,...,stream_16_estimate)<errorTol '''
    def __init__(self,distribObj,inflate=1.2,alpha=0.01,J=16,absTol=None,relTol=None,nInit=None,nMax=None):
        discDistAllowed = ["QuasiRandom"] # which discrete distributions are supported
        super().__init__(distribObj,discDistAllowed,absTol=absTol,relTol=relTol,nInit=nInit,nMax=nMax)
        self.inflate = inflate  # inflation factor
        self.alpha = alpha # uncertainty level
        self.J = J
        self.nLevels = len(distribObj)
        self.dataObj = MeanVarDataRep(self.nLevels, self.J)

    def stopYet(self,funObj):
        if self.dataObj.stage == 'begin':  # initialize
            self.dataObj._timeStart = time()  # keep track of time
            self.dataObj.prevN = zeros(self.nLevels)
            self.dataObj.nextN = full(self.nLevels,self.nInit)
            self.dataObj.costF = zeros(self.nLevels)
            self.dataObj.stage = 'sigma'
        elif self.dataObj.stage == 'sigma':
            for i in range(len(funObj)):
                if self.dataObj.sig2hat[i] < self.absTol: # Sufficient estimate for mean of funObj[i]
                    self.dataObj.flags[i] = 0
                else:
                    self.dataObj.prevN[i] = self.dataObj.nextN[i]
                    self.dataObj.nextN[i] = self.dataObj.prevN[i]*2
            if self.dataObj.flags.sum(0)==0 or self.dataObj.nextN.max() > self.nMax:
                # Stopping criterion met
                self.dataObj.solution = self.dataObj.mu2hat.sum(0)
                self.dataObj.nSamplesUsed = self.dataObj.nextN
                errBar = -norm.ppf(self.alpha / 2) * self.inflate * (self.dataObj.sig2hat**2 / self.dataObj.nextN).sum(0)**.5
                self.dataObj.confidInt = self.dataObj.solution + errBar * array([-1, 1])
                self.dataObj.stage = 'done'  # finished with computation
        self.dataObj.timeUsed = time() - self.dataObj._timeStart
        return