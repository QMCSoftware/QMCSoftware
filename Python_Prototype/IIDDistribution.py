from discreteDistribution import discreteDistribution
import numpy as np
import sys

# Specifies and generates the components of $\frac 1n \sum_{i=1}^n \delta_{\vx_i}(\cdot)$
# where the $\vx_i$ are IID uniform on $[0,1]^d$ or IID standard Gaussian
class IIDDistribution(discreteDistribution):

    def __init__(self):
        #Doctests
        """
        >>> iid = IIDDistribution()
        >>> print(iid.__dict__)
        {'domain': array([[0, 0],
               [1, 1]]), 'domainType': 'box', 'dimension': 2, 'trueDistribution': 'uniform'}
        """
        self.distribData

        super().__init__()

    @property
    def distribData(self):
        return []

    @property
    def state(self):
        return []

    @property
    def nStreams(self):
        return 1

    def initStreams(self, nStreams = 1):
        '''
        if sys._getframe().func_code.co_argcount > 1:
            self.nStreams = nStreams
        '''
        temp_streams = range(nStreams)
        #self.distribData.stream = RandStream.create('mrg32k3a', 'NumStreams', nStreams, 'CellOutput', True)
        return self

    def genDistrib(self, nStart, nEnd, n, coordIndex, streamIndex=0):
        nPts = nEnd - nStart + 1  # how many points to be generated
        if self.trueDistribution == 'uniform': # generate uniform points
            x = np.random.rand(self.distribData.stream[streamIndex], nPts, len(coordIndex))  # nodes
        else:  # standard normal points
            x = np.random.randn(self.distribData.stream[streamIndex], nPts, len(coordIndex))  # nodes
        w = 1
        a = 1 / n

        return x, w, a
    
if __name__ == "__main__":
    import doctest
    doctest.testmod()