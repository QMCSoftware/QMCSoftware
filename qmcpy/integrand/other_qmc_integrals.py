from .abstract_integrand import AbstractIntegrand
from ..true_measure import Uniform
from ..util import ParameterError
import numpy as np
import scipy

class QMCIntegrals(AbstractIntegrand):
    r"""
    Other QMC test integrals not included elsewhere, including the main 6.
    """

    def __init__(self, sampler, kind_func):
        r"""
        Args:
            sampler (AbstractDiscreteDistribution): The sampler to be used
            kind_func (str): Specifies the integrand
        """
        self.sampler = sampler
        self.true_measure = Uniform(self.sampler)
        self.d = self.true_measure.d
        self.kind_func = str(kind_func).lower().strip().replace(" ","_").replace("-","_")
        if self.kind_func not in ["sum_ueu","mc2","piece_lin_gauss","ind_sum_normal","smooth_gauss","ridge_johnson_su","ra_sum","ra_prod","ra_sin"]:
            raise ParameterError("Invalid integrand %s" % self.kind_func)
        
        self.g = self.__getattribute__(self.kind_func)
        
        super(QMCIntegrals,self).__init__(dimension_indv=(),dimension_comb=(),parallel=False)

    def sum_ueu(self,t):
        return -self.d + (t*np.exp(t)).sum(-1)
    
    def mc2(self,t):
        return -1 + ((self.d - 1/2)**(-self.d)) * np.prod(self.d - t, axis=-1)
    
    def piece_lin_gauss(self,t):
        tau = 1 #for now this is hardcoded, may be changed later
        return np.maximum(self.d**(-1/2) * scipy.stats.norm.ppf(t, 0 , 1).sum(-1) - tau, 0) - scipy.stats.norm.pdf(tau, 0, 1) + tau*scipy.stats.norm.cdf(-tau, 0, 1)
    
    def ind_sum_normal(self,t):
        tau = 1 #for now this is hardcoded, may be changed later
        return -scipy.stats.norm.pdf(-tau, 0, 1) + np.where(self.d**(-1/2) * scipy.stats.norm.ppf(t, 0 , 1).sum(-1) >= tau, 1, 0)

    def smooth_gauss(self,t):
        return -scipy.stats.norm.pdf(np.sqrt(2)/2, 0, 1) + self.d**(-1/2)*scipy.stats.norm.pdf(1 + scipy.stats.norm.ppf(t, 0, 1), 0, 1).sum(-1)
    
    def ridge_johnson_su(self,t):
        return -scipy.stats.johnsonsu.mean(1,1) + scipy.stats.johnsonsu.ppf( scipy.stats.norm.cdf(self.d**(-1/2) * scipy.stats.norm.ppf(t, 0, 1).sum(-1), 0, 1) , 1, 1)


    def ra_sum(self,t):
        return (1/self.d)*np.abs(4*t-2).sum(-1)

    def ra_prod(self,t):
        return np.prod(abs(4*t-2),axis=-1)
    
    def ra_sin(self,t):
        return np.prod(np.pi/2*np.sin(np.pi*t),axis=-1)
