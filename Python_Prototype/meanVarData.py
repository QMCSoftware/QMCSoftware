from numpy import array,zeros,std,mean
from time import time
from accumData import accumData as accumData

# Accumulated data for IIDDistribution calculations, stores the sample mean and
# variance of function values
class meanVarData(accumData):
    def __init__(self):
        super().__init__()
        self.muhat = array([]) # sample mean
        self.sighat = array([]) # sample standard deviation
        self.nSigma = array([]) # number of samples used to compute the sample standard deviation
        self.nMu = array([])  # number of samples used to compute the sample mean
        self.timeStart = time()

    @property
    def timeStart(self):  # starting time
        return self.timeStart

    def updateData(self, distribObj, funObj):
        nf = len(funObj)
        self.solution = zeros(nf)
        self.sighat = zeros(nf)
        self.costF = zeros(nf)
        for ii in range(0, nf):
            tStart = time()  # time the function values
            nStart = self.prevN[ii] + 1
            nEnd = self.prevN[ii] + self.nextN[ii]
            n = self.nextN[ii]
            coordIndex = range(0, funObj[ii].dimension)
            streamIndex = ii
            x = distribObj.genDistrib(nStart, nEnd, n, coordIndex, streamIndex)[0]
            coordIndex = range(0, funObj[ii].dimension)
            y = funObj[ii].f(x, coordIndex)
            self.costF[ii] = time() - tStart  # to be used for multi-level methods
            if self.stage == 'sigma':
                self.sighat[ii] = std(y)  # compute the sample standard deviation if required
            self.muhat[ii] = mean(y)  # compute the sample mean
            self.solution = sum(self.muhat)  # which also acts as our tentative solution
        return
    
    def __repr__(self):
        return str(self.__dict__)

if __name__ == "__main__":
    # Doctests
    import doctest
    x = doctest.testfile("Tests/dt_meanVarData.py")
    print("\n"+str(x))
