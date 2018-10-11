from discreteDistribution import discreteDistribution as discreteDistribution
import randomstate.prng.mrg32k3a as rnd

# Specifies and generates the components of $\frac 1n \sum_{i=1}^n \delta_{\vx_i}(\cdot)$
# where the $\vx_i$ are IIDDistribution uniform on $[0,1]^d$ or IIDDistribution standard Gaussian
class IIDDistribution(discreteDistribution):

    def __init__(self):
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

    def __eq__(self,obj1): #automaticlly gets called for "IIDobj1 == IIDobj2"
        c1 = self.domain == obj1.domain
        c2 = self.domainType == obj1.domainType
        c3 = self.dimension == obj1.dimension
        c4 = self.trueDistribution == obj1.trueDistribution
        if c1 and c2 and c3 and c4:
            return True
        return False

    # May need to implement this differently as we currently cannot compare 2 obj's of this class without a seperate instance of this class
    def compare2Objs(self,obj1,obj2): 
        c1 = obj1.domain = obj1.domain
        c2 = obj1.domainType == obj1.domainType
        c3 = obj1.dimension == obj1.dimension
        c4 = obj1.trueDistribution == obj1.trueDistribution
        if c1 and c2 and c3 and c4:
            return True
        return False

    def initStreams(self, nStreams = 1, seed=None):
        self.distribDataStream = list(range(nStreams))
        for i in range(nStreams):
            self.distribDataStream[i] = rnd.RandomState(seed)
        return self

    def genDistrib(self, nStart, nEnd, n, coordIndex, streamIndex=0):
        nPts = nEnd - nStart + 1  # how many points to be generated

        if self.trueDistribution == 'uniform': # generate uniform points
            x = self.distribDataStream[streamIndex].rand(int(nPts), len(coordIndex))  # nodes
        else:  # standard normal points
            x = self.distribDataStream[streamIndex].randn(int(nPts), len(coordIndex))  # nodes

        # Code without streams
        w = 1
        a = 1 / n
        return x, w, a

if __name__ == "__main__":
    import doctest
    x = doctest.testfile("dt_IID.py")
    print("\n"+str(x))
