from abc import ABC, abstractmethod
import numpy as np

# Specifies and generates the components of §$\mcommentfont a_n \sum_{i=1}^n w_i \delta_{\vx_i}(\cdot)$
class discreteDistribution(ABC):
    '''
    Any sublcass of discreteDistribution must include:
        Properties: distribData, state, nStreams
        Methods: genDistrib(self, nStart, nEnd, n, coordIndex)
    '''
    distribObjs = []
    def __init__(self,trueD):
        super().__init__()
        self.trueD = trueD # the distribution that the discrete distribution attempts to emulate
        discreteDistribution.distribObjs.append(self)
    # Abstract Properties
    @property
    @abstractmethod
    def distribData(self):  # %information required to generate the distribution
        pass

    @property
    @abstractmethod
    def state(self):  # state of the generator
        pass

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
        return len(discreteDistribution.distribObjs)
    def __iter__(self):
        for distribObj in discreteDistribution.distribObjs:
            yield distribObj
    def __getitem__(self,i):
        return discreteDistribution.distribObjs[i]

