from numpy import exp, maximum

from . import Integrand

class AsianCallFun(Integrand):
    ''' Specify and generate payoff values of an Asian Call option'''
    def __init__(self, BMmeasure=None, volatility=.5, S0=30, K=25, nominal_value=None):
        super().__init__(nominal_value=nominal_value)
        self.BMmeasure = BMmeasure
        self.volatility = volatility
        self.S0 = S0
        self.K = K
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
        SFine = self.S0*exp((-self.volatility**2/2)*self.BMmeasure.timeVector+self.volatility*x)
        AvgFine = ((self.S0/2)+SFine[:,:self.dimension-1].sum(1)+SFine[:,self.dimension-1]/2)/self.dimension
        y = maximum(AvgFine-self.K,0)
        if self.dimFac > 0:
            Scourse = SFine[:,int(self.dimFac-1)::int(self.dimFac)]
            dCourse = self.dimension/self.dimFac
            AvgCourse = ((self.S0/2)+Scourse[:,:int(dCourse)-1].sum(1)+Scourse[:,int(dCourse)-1]/2)/dCourse
            y = y-maximum(AvgCourse-self.K,0)
        return y
