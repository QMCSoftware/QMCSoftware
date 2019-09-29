''' Originally developed in MATLAB by Fred Hickernell. Translated to python by Sou-Cheng T. Choi and Aleksei Sorokin '''
from numpy import random

from . import DiscreteDistribution

class IIDDistribution(DiscreteDistribution):
    '''
    Specifies and generates the components of $\frac 1n \sum_{i=1}^n \delta_{\vx_i}(\cdot)$
    where the $\vx_i$ are IIDDistribution uniform on $[0,1]^d$ or IIDDistribution standard Gaussian
    '''
    
    def __init__(self,trueD=None,distribData=None,rngSeed=None):
        accepted_measures = ['stdUniform','stdGaussian']
        if rngSeed: random.seed(rngSeed)
        super().__init__(accepted_measures,trueD,distribData)

    def genDistrib(self,n,m,j=1):
        if self.trueD.measureName=='stdUniform': return random.rand(j,int(n),int(m)).squeeze()
        elif self.trueD.measureName=='stdGaussian': return random.randn(j,int(n),int(m)).squeeze()