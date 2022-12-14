from ._accumulate_data import AccumulateData
from numpy import *

class MeanVarDataRep(AccumulateData):
    """
    Update and store mean and variance estimates with repliations. 
    See the stopping criterion that utilize this object for references.
    """

    def __init__(self, z_star, inflate, replications):
        self.z_star = z_star
        self.inflate = inflate
        self.replications = replications
        self.ysums = zeros(self.replications,dtype=float)
        self.n_rep = 0
        super(MeanVarDataRep,self).__init__()

    def update_data(self,y):
        self.n_rep += len(y)
        self.ysums += y.sum(0)
        self.muhats = self.ysums/self.n_rep
        self.muhat = self.muhats.mean()
        self.sighat = self.muhats.std()
        self.bounds = self.muhat+array([-1,1])*self.z_star*self.inflate*self.sighat/sqrt(self.replications)
        return self.muhat,self.bounds[0],self.bounds[1]
