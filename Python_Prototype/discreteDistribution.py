from abc import ABC, abstractmethod
import numpy as np

# Specifies and generates the components of §$\mcommentfont a_n \sum_{i=1}^n w_i \delta_{\vx_i}(\cdot)$§
class discreteDistribution(ABC):

    def __init__(self, value):
        self.domain = np.array([[0, 0], [1, 1]])  # domain of the discrete distribution, §$\mcommentfont \cx$§
        self.domainType = 'box'  # domain of the discrete distribution, §$\mcommentfont \cx$§
        self.dimension = 2  # dimension of the domain, §$\mcommentfont d$§
        self.trueDistribution = 'uniform'  # name of the distribution that the discrete distribution attempts to emulate
        super().__init__()

    @abstractmethod
    def distribData(self):  # %information required to generate the distribution
        pass

    @abstractmethod
    def state(self):  # state of the generator
        pass

    @abstractmethod
    def nStreams(self):
        pass

    # Abstract Method
    @abstractmethod
    def genDistrib(self, nStart, nEnd, n, coordIndex):
        """
         nStart = starting value of §$\mcommentfont i$§
         nEnd = ending value of §$\mcommentfont i$§
         n = value of §$\mcommentfont n$§ used to determine §$\mcommentfont a_n$§
         coordIndex = which coordinates in sequence are needed
        """
        pass
