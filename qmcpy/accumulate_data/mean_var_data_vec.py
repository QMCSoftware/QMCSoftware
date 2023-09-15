from ._accumulate_data import AccumulateData
from numpy import *

class MeanVarDataVec(AccumulateData):
    """
    Recompute mean and variance estimates for an iterative double sample size. 
    See the stopping criterion that utilize this object for references.
    """

    def __init__(self, z_star, inflate):
        self.z_star = z_star
        self.inflate = inflate
        super(MeanVarDataVec,self).__init__()

    def update_data(self,y):
        muhat = y.mean()
        sighat = y.std()
        bounds = muhat+array([-1,1])*self.z_star*self.inflate*sighat/sqrt(len(y))
        return muhat,bounds[0],bounds[1]