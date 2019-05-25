from discreteDistribution import discreteDistribution as discreteDistribution
from randomstate.prng import mrg32k3a
from numpy import arange

class IIDDistribution(discreteDistribution):
    '''
    # Specifies and generates the components of $\frac 1n \sum_{i=1}^n \delta_{\vx_i}(\cdot)$
    # where the $\vx_i$ are IIDDistribution uniform on $[0,1]^d$ or IIDDistribution standard Gaussian
    '''

    def __init__(self,trueD):
        super().__init__(trueD)
        self.state = [] # not used
        self.distribData = distribData # stream data
        if (not self.distribData) or (not self.distribData.shape==(1,len(self))):
            raise Exception("distribData must have shape"+str(self.distribData.shape)+"to match the list of functions") 
        
    @property
    def distribData(self):
        return self.distribData
    @property
    def state(self):
        return self.state

    def initStreams(self,seed=13):
        nObj = len(self)
        for ii in range(nObj):
            self[ii].distribData.stream = mrg32k3a.RandomState(seed)
        return self

    def genDistrib(self, nStart, nEnd, n, coordIndex=arange(1,self.trueD.dimension+1)):
        nPts = nEnd - nStart + 1  # how many points to be generated
        if self.trueDistribution=='stdUniform': # generate uniform points
            x = self.distribData.stream.rand(int(nPts),len(coordIndex))  # nodes
        elif self.trueD.measureName=='stdGaussian': # standard normal points
            x = self.distribData.stream.randn(int(nPts),len(coordIndex))  # nodes
        else:
            raise Exception('Distribution not recognized')
        return x,1,1/n

    @staticmethod
    def compare2Objs(obj1,obj2): 
        if obj1.distribData==obj2.distribData and obj1.state==obj2.state:
            return True
        return False

    def __eq__(self,obj1):
        ''' automaticlly gets called for "IIDobj1 == IIDobj2" '''
        if self.distribData==obj1.distribData and self.state==obj1.state:
            return True
        return False

if __name__ == "__main__":
    import doctest
    x = doctest.testfile("Tests/dt_IIDDistribution.py")
    print("\n"+str(x))
