from ._integrand import Integrand
from ..util import ParameterError
from ..true_measure import Uniform
from numpy import *


class SobolIndices(Integrand):
    """
    Sobol' Indicies in QMCPy. 
    
    References: 
        [1] Art B. Owen.Monte Carlo theory, methods and examples. 2013. Appendix A.
    """
    def __init__(self, integrand, indices='singletons'):
        """
        Args:
            integrand (Integrand): integrand to find Sobol' indices of
            indices (list of lists): each element of indices should be a list of indices, u,
                at which to compute the Sobol' indices. 
                The default indices='singletons' sets indices=[[0],[1],...[d-1]]. 
                Should not include [], corresponding to the null set
        """
        self.parameters = []
        self.integrand = integrand
        self.d = self.integrand.d
        # indices
        if self.integrand.dprime>1: 
            raise ParameterError('SobolIndices currently onlly supports integrand.dprime = 1')
        self.indices = indices
        if self.indices=='singletons':
            self.indices = [[j] for j in range(self.d)]
        if [] in self.indices:
            raise ParameterError('SobolIndices indices cannot include [], the null set.')
        self.indices_sizes = array([len(u) for u in self.indices])
        self.s = len(self.indices)
        self.indices_bool_mat = tile(False,(self.s,self.d))
        for k in range(self.s): self.indices_bool_mat[k,self.indices[k]] = True
        self.not_indices_bool_mat = ~self.indices_bool_mat
        # sensitivity_index
        self.dtilde = 2*self.d
        self.dprime = 2*self.s+2
        self.true_measure = self.integrand.true_measure
        self.discrete_distrib = self.true_measure.discrete_distrib.spawn(s=1,dimensions=[self.dtilde])[0]
        self.sampler = self.integrand.sampler
        self.g = self.integrand.g
        super(SobolIndices,self).__init__()
    
    def f(self, x, periodization_transform='NONE', *args, **kwargs):
        z = x[:,self.d:]
        x = x[:,:self.d]
        y = zeros((x.shape[0],self.dprime),dtype=float)
        if 'compute_flags' in kwargs:
            compute_flags = kwargs['compute_flags']
            del kwargs['compute_flags']
        else:
            compute_flags = ones(self.dprime)
        v = zeros(x.shape,dtype=float)
        f_x = self.integrand.f(x,periodization_transform,*args,**kwargs).squeeze()
        f_z = self.integrand.f(z,periodization_transform,*args,**kwargs).squeeze()
        for k in range(self.s):
            u_bool = self.indices_bool_mat[k]
            not_u_bool = self.not_indices_bool_mat[k]
            v[:,u_bool] = x[:,u_bool]
            v[:,not_u_bool] = z[:,not_u_bool]
            f_v = self.integrand.f(v,periodization_transform,*args,**kwargs).squeeze()
            y[:,k] = f_z*(f_x-f_v) # closed
            y[:,self.s+k] = (f_z-f_v)**2/2 # total
        y[:,-2] = f_x # mu
        y[:,-1] = f_x**2 # sigma^2+mu^2
        return y
    
    def _spawn(self, level, sampler):
        new_integrand = self.integrand.spawn(level,sampler)
        return SobolIndices(
            integrand = new_integrand,
            indices = self.indices)