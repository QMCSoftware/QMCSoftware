''' Originally developed in MATLAB by Fred Hickernell. Translated to python by Sou-Cheng T. Choi and Aleksei Sorokin '''
from numpy.random import rand, randn

from . import discreteDistribution

class IIDDistribution(discreteDistribution):
    '''
    Specifies and generates the components of $\frac 1n \sum_{i=1}^n \delta_{\vx_i}(\cdot)$
    where the $\vx_i$ are IIDDistribution uniform on $[0,1]^d$ or IIDDistribution standard Gaussian
    '''
    
    def __init__(self,distribData=None,trueD=None):
        state = []
        super().__init__(distribData,state,trueD=trueD)
        if trueD:
            self.distrib_list = [IIDDistribution() for i in range(len(trueD))]
            # self now refers to self.distrib_list
            for i in range(len(self)):
                self[i].trueD = self.trueD[i]
                self[i].distribData = self.distribData[i] if self.distribData else None

    def genDistrib(self,n,m,j=1):
        if self.trueD.measureName=='stdUniform': # generate uniform points
            return rand(j,int(n),int(m)).squeeze()
        elif self.trueD.measureName=='stdGaussian': # standard normal points
            return randn(j,int(n),int(m)).squeeze()
        else:
            raise Exception('Distribution not recognized')
