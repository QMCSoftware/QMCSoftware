import numpy as np
from time import time
from Accumulate_Data import Accumulate_Data

# Accumulated data for IID calculations, stores the sample mean and
# variance of function values
class meanVar(Accumulate_Data):


    def __init__(self):
        self.muhat = [] # sample mean
        self.sighat = [] # sample standard deviation
        self.nSigma = [] # number of samples used to compute the sample standard deviation
        self.nMu = []  # number of samples used to compute the sample mean
        super().__init__()


    def timeStart(self):  # starting time
        self.__timeStart = time()
        #print(self.__timeStart) # "hidden" property, but it can still be exposed somehow as the last doctest shows
        return


    def updateData(self, distribObj, funObj):
        nf = 1
        if type(funObj) == list:
            nf = len(funObj)
        else:
            funObj = [funObj]
        # preallocate vectors
        self.solution = np.zeros(nf)
        self.sighat = np.zeros(nf)
        self.costF = np.zeros(nf)
        for ii in range(0, nf):
            tStart = time()  # time the function values
            y = funObj[ii].f(distribObj.genDistrib(self.prevN[ii] + 1, self.prevN[ii] + self.nextN[ii], self.nextN[ii],
                                               range(0, funObj[ii].dimension), ii)[0],
                         range(0, funObj[ii].dimension))
            self.costF[ii] = time() - tStart  # to be used for multi-level methods
            if self.stage == 'sigma':
                self.sighat[ii] = np.std(y)  # compute the sample standard deviation if required
            self.muhat[ii] = np.mean(y)  # compute the sample mean
            self.solution = sum(self.muhat)  # which also acts as our tentative solution

        return self

if __name__ == "__main__":
    # Doctests
    '''
    import doctest
    x = doctest.testfile("dt_meanVarData.py")
    print("\n"+str(x))
    '''
    print('blah')