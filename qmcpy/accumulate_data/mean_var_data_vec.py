from ._accumulate_data import AccumulateData
import numpy as np

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
        sighat = y.std(ddof = 1)
        bounds = muhat+np.array([-1,1])*self.z_star*self.inflate*sighat/np.sqrt(len(y))
        return muhat,bounds[0],bounds[1]