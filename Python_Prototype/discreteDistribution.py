from abc import ABC, abstractmethod
import numpy as np

from measure import measure

class discreteDistribution(ABC):
    '''
    Specifies and generates the components of §$\mcommentfont a_n \sum_{i=1}^n w_i \delta_{\vx_i}(\cdot)$
    '''
    def __init__(self,distribData,state,trueD=measure('stdUniform')):
        super().__init__()
        # Abstract Properties
        self.distribData = distribData # information required to generate the distribution
        self.state = state # state of the generator

        self.trueD = trueD      
        self.distrib_list = []

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
    
    # Below methods allow the distribution class to be treated like a list of distributions
    def __len__(self):
        return len(self.distrib_list)
    def __iter__(self):
        for distribObj in self.distrib_lists:
            yield distribObj
    def __getitem__(self,i):
        return self.distrib_list[i]
    
    def __repr__(self):
        s = str(type(self).__name__)+' with properties:\n'
        for key,val in self.__dict__.items():
            s += '    %s: %s\n'%(str(key),str(val))
        return s[:-1]

