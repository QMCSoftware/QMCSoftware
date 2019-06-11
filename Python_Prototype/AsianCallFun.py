''' Originally developed in MATLAB by Fred Hickernell. Translated to python by Sou-Cheng T. Choi and Aleksei Sorokin '''
from fun import fun as fun
from numpy import square, cos, exp, sqrt, multiply, sum, array, cumprod, transpose, ones,maximum
from numpy.linalg import eig
from scipy.sparse import spdiags

class AsianCallFun(fun):
    ''' Specify and generate payoff values of an Asian Call option'''
    def __init__(self,BMmeasure=None):
        super().__init__()
        self.volatility = 0.5
        self.S0 = 30
        self.K = 25
        self.BMmeasure = BMmeasure
        self.dimFac = 0

        if self.BMmeasure:
            nBM = len(BMmeasure)
            self.fun_list = [AsianCallFun() for i in range(nBM)] 
            self[0].BMmeasure = self.BMmeasure[0]
            self[0].dimFac = 0
            self[0].dimension = self.BMmeasure[0].dimension
            for ii in range(1,nBM):
                self[ii].BMmeasure = self.BMmeasure[ii]
                self[ii].dimFac = self.BMmeasure[ii].dimension/self.BMmeasure[ii-1].dimension
                self[ii].dimension = self.BMmeasure[ii].dimension  

    def g(self,x,ignore):
        SFine = self.S0*exp((-self.volatility**2/2)*self.BMmeasure.measureData['timeVector']+self.volatility*x)
        AvgFine = ((self.S0/2)+SFine[:,:self.dimension-1].sum(1)+SFine[:,self.dimension-1]/2)/self.dimension
        y = maximum(AvgFine-self.K,0)
        if self.dimFac > 0:
            Scourse = SFine[:,int(self.dimFac-1)::int(self.dimFac)]
            dCourse = self.dimension/self.dimFac
            AvgCourse = ((self.S0/2)+Scourse[:,:int(dCourse)-1].sum(1)+Scourse[:,int(dCourse)-1]/2)/dCourse
            y = y-maximum(AvgCourse-self.K,0)
        return y

if __name__ == "__main__":
    # Run Doctests
    import doctest
    x = doctest.testfile("Tests/dt_AsianCallFun.py")
    print("\n" + str(x))

