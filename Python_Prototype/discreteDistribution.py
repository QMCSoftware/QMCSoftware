''' Originally developed in MATLAB by Fred Hickernell. Translated to python by Sou-Cheng T. Choi and Aleksei Sorokin '''
from abc import ABC, abstractmethod
import numpy as np
from util import univ_repr

from measure import measure

class discreteDistribution(ABC):
    '''
    Specifies and generates the components of §$\mcommentfont a_n \sum_{i=1}^n w_i \delta_{\vx_i}(\cdot)$
        Any sublcass of discreteDistribution must include:
            Methods: genDistrib(self,nStart,nEnd,n,coordIndex)
            Properties: distribData,state,trueD
    '''
    def __init__(self,distribData,state,trueD=None):
        super().__init__()
        # Abstract Properties
        self.distribData = distribData # information required to generate the distribution
        self.state = state # state of the generator
        self.trueD = trueD if trueD else measure().stdUniform()   
        self.distrib_list = [self]

    # Abstract Methods
    @abstractmethod
    def genDistrib(self, nStart, nEnd, n, coordIndex):
        """
         nStart = starting value of §$\mcommentfont i$§
         nEnd = ending value of §$\mcommentfont i$§
         n = value of §$\mcommentfont n$§ used to determine §$\mcommentfont a_n$§
         coordIndex = which coordinates in sequence are needed
        """
        pass
    
    # Magic Methods. Makes self[i]==self.distrib_list[i]
    def __len__(self): return len(self.distrib_list)
    def __iter__(self):
        for distribObj in self.distrib_list:
            yield distribObj
    def __getitem__(self,i): return self.distrib_list[i]
    def __setitem__(self,i,val): self.distrib_list[i]=val
    def __repr__(self): return univ_repr(self,'discreteDistribution','distrib_list')

