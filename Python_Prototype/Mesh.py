''' Originally developed in MATLAB by Fred Hickernell. Translated to python by Sou-Cheng T. Choi and Aleksei Sorokin '''
from discreteDistribution import discreteDistribution
from numpy import arange,log,array
from numpy.random import rand,randn
# Import random shift functions
from latticeseq_b2 import get_RS_lattice_b2
from digitalsequence_b2g import get_RS_sobol_b2g

class Mesh(discreteDistribution):

    def __init__(self,distribData=None,trueD=None):
        state = []
        super().__init__(distribData,state,trueD=trueD)
        if trueD:
            self.distrib_list = [Mesh() for i in range(len(trueD))]
            # self now refers to self.distrib_list
            for i in range(len(self)):
                self[i].trueD = self.trueD[i]
                self[i].distribData = self.distribData[i] if self.distribData else None

    def genDistrib(self,n,m,j=1):
        # get j randomly shifted nxm arrays 
        mimicMeasure = self.trueD.measureName
        if mimicMeasure=='stdUniform':
            meshType = self.trueD.measureData['meshType']
            if meshType=='lattice': return get_RS_lattice_b2(n,m,j)
            elif meshType=='sobol': return get_RS_sobol_b2g(n,m,j)
            else: raise Exception("%s mesh cannot mimic %s distribution"%(meshType,mimicMeasure))
        else: raise Exception('Distribution not recognized')

if __name__ == "__main__":
    import doctest
    x = doctest.testfile("Tests/dt_IIDDistribution.py")
    print("\n"+str(x))
