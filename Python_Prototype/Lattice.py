''' Originally developed in MATLAB by Fred Hickernell. Translated to python by Sou-Cheng T. Choi and Aleksei Sorokin '''
from discreteDistribution import discreteDistribution
from numpy import arange,log
from numpy.random import rand,randn
from latticeseq_b2 import get_lattice_b2

class Lattice(discreteDistribution):

    def __init__(self,distribData=None,trueD=None):
        state = []
        super().__init__(distribData,state,trueD=trueD)
        if trueD:
            self.distrib_list = [Lattice() for i in range(len(trueD))]
            # self now refers to self.distrib_list
            for i in range(len(self)):
                self[i].trueD = self.trueD[i]
                self[i].distribData = self.distribData[i] if self.distribData else None

    def genDistrib(self,n,m,j=1):
        if self.trueD.measureName=='stdUniform':
            x = get_lattice_b2(int(log(n)/log(2)),int(m))
            return [(x+rand(m))%1 for i in range(j)]
        else:
            raise Exception('Distribution not recognized')

if __name__ == "__main__":
    import doctest
    x = doctest.testfile("Tests/dt_IIDDistribution.py")
    print("\n"+str(x))
