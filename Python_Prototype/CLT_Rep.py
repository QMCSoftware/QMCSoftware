''' Originally developed in MATLAB by Fred Hickernell. Translated to python by Sou-Cheng T. Choi and Aleksei Sorokin '''
from numpy import array,zeros,floor,log10,tile,full,inf,log,ones
from time import time
from scipy.stats import norm

from meanVarData_Rep import meanVarData_Rep
from stoppingCriterion import stoppingCriterion

class CLT_Rep(stoppingCriterion):
    ''' Stopping criterion based on var(stream_1_estimate,stream_2_estimate,...,stream_16_estimate)<errorTol '''
    def __init__(self,inflate=1.2,alpha=0.01,J=16,absTol=None,relTol=None,nInit=None,nMax=None):
        discDistAllowed = ["IIDDistribution"] # which discrete distributions are supported
        decompTypeAllowed = ["single", "multi"] # which decomposition types are supported
        super().__init__(discDistAllowed,decompTypeAllowed,absTol=absTol,relTol=relTol,nInit=nInit,nMax=nMax)
        self.inflate = inflate  # inflation factor
        self.alpha = alpha # uncertainty level
        self.J = J

    def stopYet(self,dataObj=None,funObj=[],distribObj=[]):
        nf=len(funObj)
        if dataObj==None: dataObj=meanVarData_Rep(nf,self.J)
        if dataObj.stage == 'begin':  # initialize
            dataObj._timeStart = time()  # keep track of time
            if type(distribObj).__name__ not in self.discDistAllowed:
                raise Exception('Stopping criterion not compatible with sampling distribution')
            dataObj.prevN = zeros(nf)
            dataObj.nextN = full(nf,self.nInit)
            dataObj.costF = zeros(nf)
            dataObj.stage = 'sigma'
        elif dataObj.stage == 'sigma':
            if dataObj.prevN.max()*2 > self.nMax: # Do not need to calculate dataObj.nextN
                dataObj.stage = 'mu'
            else: # update dataObj.nextN and flag functions whose means have been sufficiently estimated
                dataObj.prevN = dataObj.nextN
                for i in range(len(funObj)):
                    if dataObj.sig2hat[i]<self.absTol: # Sufficient estimate for mean of funObj[i]
                        dataObj.flags[i] = 0
                    else:
                        dataObj.nextN[i] = dataObj.prevN[i]*2
                if dataObj.flags.sum(0)==0:
                    dataObj.stage = 'mu'
        elif dataObj.stage == 'mu':
            dataObj.solution = dataObj.mu2hat.sum(0)
            dataObj.nSamplesUsed = dataObj.nextN
            errBar = self.get_quantile() * self.inflate * (dataObj.sig2hat**2 / dataObj.nextN).sum(0)**.5 # Correct?
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


    # Need to encorporate (Delete later)
    self.solution = nan  # solution
    self.stage = 'begin'  # stage of the computation, becomes 'done' when finished
    self.prevN = array([])  # new data will be based on (quasi-)random vectors indexed by
    self.nextN = array([])  # prevN + 1 to nextN
    self.timeUsed = array([])  # time used so far
    self.nSamplesUsed = array([])  # number of samples used so far
    self.confidInt = array([-inf, inf])  # error bound on the solution
    self.costF = array([])  # time required to compute function values
    self._timeStart = None # hidden/private