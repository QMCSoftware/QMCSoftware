from ._accumulate_data import AccumulateData
import numpy as np

class MeanVarDataRep(AccumulateData):
    """
    Update and store mean and variance estimates with replications. 
    See the stopping criterion that utilize this object for references.
    """

    def __init__(self, stopping_crit):
        self.parameters = [
            'solution',
            'comb_bound_low',
            'comb_bound_high',
            'comb_bound_diff',
            'comb_flags',
            'n_total',
            'n',
            'n_rep',
            'time_integrate']
        self.stopping_crit = stopping_crit
        self.integrand = self.stopping_crit.integrand
        self.true_measure = self.integrand.true_measure
        self.discrete_distrib = self.true_measure.discrete_distrib
        self.flags_indv = np.tile(False,self.integrand.d_indv)
        self.compute_flags = np.tile(True,self.integrand.d_indv)
        self.n_rep = np.tile(self.stopping_crit.n_init,self.integrand.d_indv)
        self.n_min = 0
        self.n_max = int(self.n_rep.max())
        self.solution_indv = np.tile(np.nan,self.integrand.d_indv)
        self.xfull = np.empty((self.discrete_distrib.replications,0,self.integrand.d))
        self.yfull = np.empty(self.integrand.d_indv+(self.discrete_distrib.replications,0))
        self._ysums = np.zeros(self.integrand.d_indv+(self.discrete_distrib.replications,),dtype=float)
        self.ns = np.array([0],dtype=int)
        super(MeanVarDataRep,self).__init__()

    def update_data(self):
        xnext = self.discrete_distrib(n_min=self.n_min,n_max=self.n_max)
        ynext = self.integrand.f(xnext,compute_flags=self.compute_flags)
        ynext[~self.compute_flags] = np.nan
        self.ns = np.append(self.ns,self.n_max)
        self.xfull = np.concatenate([self.xfull,xnext],1)
        self.yfull = np.concatenate([self.yfull,ynext],-1)
        self.n_rep[self.compute_flags] = self.n_max
        self._ysums[self.compute_flags] += ynext[self.compute_flags].sum(-1)
        self.muhats = self._ysums/self.n_rep[...,None]
        self.solution_indv = self.muhats.mean(-1)
        self.sigmahat = self.muhats.std(-1)
        self.ci_half_width = self.stopping_crit.t_star*self.stopping_crit.inflate*self.sigmahat/np.sqrt(self.discrete_distrib.replications)
        self.indv_bound_low = self.solution_indv-self.ci_half_width
        self.indv_bound_high = self.solution_indv+self.ci_half_width
        self.n = self.discrete_distrib.replications*self.n_rep
        self.n_total = self.n.max() 
        self.n_min = self.n_max
        self.n_max = 2*self.n_min
        
