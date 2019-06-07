from numpy import array,zeros,std,mean,arange
from time import clock
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
        if not self.solution: self.solution = zeros(nf)
        if not self.muhat: self.muhat = zeros(nf)
        if not self.sighat: self.sighat = zeros(nf)
        if not self.costF: self.costF = zeros(nf)
        for ii in range(nf):
            tStart = clock()  # time the function values
            dim = distribObj[ii].trueD.dimension
            distribData,*ignore = distribObj[ii].genDistrib(self.prevN[ii]+1,self.prevN[ii]+self.nextN[ii],self.nextN[ii],arange(1,dim+1))
            y = funObj[ii].f(distribData,arange(1,dim+1))
            self.costF[ii] = clock() - tStart  # to be used for multi-level methods
            if self.stage == 'sigma':
                self.sighat[ii] = std(y)  # compute the sample standard deviation if required
            self.muhat[ii] = y.mean(0)  # compute the sample mean
            self.solution = self.muhat.sum(0) # which also acts as our tentative solution
        return self

if __name__ == "__main__":
    # Doctests
    import doctest
    x = doctest.testfile("Tests/dt_meanVarData.py")
    print("\n"+str(x))
