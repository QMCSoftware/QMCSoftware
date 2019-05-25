from numpy import array,zeros,std,mean,arange
from time import time
from accumData import accumData as accumData


class meanVarData(accumData):
    '''
    Accumulated data for IIDDistribution calculations,
    stores the sample mean and variance of function values
    '''
    def __init__(self):
        super().__init__()
        self.muhat = None # sample mean
        self.sighat = None # sample standard deviation
        self.nSigma = None # number of samples used to compute the sample standard deviation
        self.nMu = None  # number of samples used to compute the sample mean

    def updateData(self, distribObj, funObj):
        nf = len(funObj)
        self.solution = zeros(nf)
        self.muhat = zeros(nf)
        self.sighat = zeros(nf)
        self.costF = zeros(nf)
        for ii in range(0, nf):
            tStart = time()  # time the function values
            dim = distribObj[ii].trueD.dimension
            this_distrib = genDistrib(distribObj[ii],self.prevN[ii]+1,self.prevN[ii]+obj.nextN[ii],self.nextN[ii],arange(1,dim+1))
            y = funObj[ii].f(this_distrib,arange(1,dim+1))
            self.costF[ii] = time() - tStart  # to be used for multi-level methods
            if self.stage == 'sigma':
                self.sighat[ii] = std(y)  # compute the sample standard deviation if required
            self.muhat[ii] = mean(y)  # compute the sample mean
            self.solution = self.muhat.sum(0) # which also acts as our tentative solution
        return self
    
    def __repr__(self):
        return str(self.__dict__)

if __name__ == "__main__":
    # Doctests
    import doctest
    x = doctest.testfile("Tests/dt_meanVarData.py")
    print("\n"+str(x))
