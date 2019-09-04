''' Originally developed in MATLAB by Fred Hickernell. Translated to python by Sou-Cheng T. Choi and Aleksei Sorokin '''
from numpy import zeros, ones, arange
from time import process_time
eps = finfo(float32).eps

from algorithms.accumData.accumData import accumData

class meanVarData_Rep(accumData):
    ''' Accumulated data for lattice calculations '''
    def __init__(self,nf,J):
        '''
        nf = # function
        J = # streams
        '''
        super().__init__()
        self.J = J
        self.muhat = zeros(self.J)
        self.mu2hat = zeros(nf)
        self.sig2hat = zeros(nf)
        self.flags = ones(nf)
        
    def updateData(self, distribObj, funObj):
        for i in range(len(funObj)):
            if self.flags[i]==0: # mean of funObj[i] already sufficiently estimated
                continue
            tStart = process_time()  # time the function values
            dim = distribObj[i].trueD.dimension
            set_x = distribObj[i].genDistrib(self.nextN[i],dim,self.J) # set of j distribData_{nxm}
            for j in range(self.J):
                y = funObj[i].f(set_x[j],arange(1,dim+1))
                self.muhat[j] = y.mean(0)
            self.costF[i] = max(process_time()-tStart,eps) 
            self.mu2hat[i] = self.muhat.mean(0)
            self.sig2hat[i] = self.muhat.std(0)
        self.solution = self.mu2hat.sum(0)
        return self

if __name__ == "__main__":
    # Doctests
    print('Still need to write doctest for this')
