from .abstract_integrand import AbstractIntegrand
from ..true_measure import Uniform
from ..util import ParameterError
import numpy as np

class QMCIntegrals(AbstractIntegrand):
    r"""
    Other QMC test integrals not included elsewhere
    """

    def __init__(self, sampler, kind_func):
        r"""
        Args:
            
        """
        self.sampler = sampler
        self.sampler = sampler
        self.true_measure = Uniform(self.sampler)
        self.d = self.true_measure.d
        self.kind_func = str(kind_func).upper().strip().replace("_"," ").replace("-"," ")
        if self.kind_func not in ["SUM UEU","RA SUM","RA PROD","RA SIN"]:
            raise ParameterError("Invalid integrand %s" % self.kind_func)
        if self.kind_func=="SUM UEU":
            self.g = self.sum_UeU
        elif self.kind_func=="RA SUM":
            self.g = self.raSum
        elif self.kind_func=="RA PROD":
            self.g = self.raProd
        elif self.kind_func=="RA SIN":
            self.g = self.raSin
        super(QMCIntegrals,self).__init__(dimension_indv=(),dimension_comb=(),parallel=False)

    def sum_UeU(self,t):
        return self.d + (t*np.exp(t)).sum(-1)
    
    def raSum(self,t):
        return (1/self.d)*np.abs(4*t-2).sum(-1)

    def raProd(self,t):
        return np.prod(abs(4*t-2),axis=-1)
    
    def raSin(self,t):
        return np.prod(np.pi/2*np.sin(np.pi*t),axis=-1)
