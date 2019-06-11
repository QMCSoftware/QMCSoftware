''' Originally developed in MATLAB by Fred Hickernell. Translated to python by Sou-Cheng T. Choi and Aleksei Sorokin '''
from numpy import array,zeros,floor,log10,tile,full,inf
from time import time
from scipy.stats import norm

from meanVarData import meanVarData
from stoppingCriterion import stoppingCriterion

class CompVarStopping(stoppingCriterion):
    ''' Stopping criterion based on var(stream_1_estimate,stream_2_estimate,...,stream_16_estimate)<errorTol '''
    def __init__(self):
        discDistAllowed = ["IIDDistribution"]
        decompTypeAllowed = ["single", "multi"]
        super().__init__(discDistAllowed,decompTypeAllowed)
        self.nInit = 1000
        self.inflate = 1.2
        self.alpha = 0.01
        
    def stopYet(self,dataObj=None,funObj=[],distribObj=[]):
        if dataObj==None: dataObj=meanVarData()
        if dataObj.stage == 'begin':  # initialize
            dataObj._timeStart = time()  # keep track of time
            if type(distribObj).__name__ not in self.discDistAllowed:
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
            if any(dataObj.sighat) > self.absTol and dataObj.prevN*10 <= self.nMax: # try again with 10x more samples
                dataObj.nextN = dataObj.prevN*10
            else:
                dataObj.stage = 'mu'
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
    print('Still need to write doctest for this')