from ._integrand import Integrand
from . import Keister
from ..stopping_criterion import CubQMCCLT
from ..util import ParameterError
from ..true_measure import Uniform
from ..discrete_distribution import DigitalNetB2
from numpy import *


class SobolIndices(Integrand):
    """
    Sobol' Indicies in QMCPy. 

    >>> dnb2 = DigitalNetB2(dimension=3,seed=7)
    >>> keister_d = Keister(dnb2)
    >>> keister_indices = SobolIndices(keister_d,indices='singletons')
    >>> sc = CubQMCCLT(keister_indices,abs_tol=.05)
    >>> solution,data = sc.integrate()
    >>> data
    MeanVarDataRep (AccumulateData Object)
        solution        [0.321 0.331 0.328 0.334 0.341 0.334]
        indv_error      [0.092 0.103 0.046 0.048 0.072 0.058 0.007 0.051]
        ci_low          [1.555 1.595 1.64  1.667 1.679 1.659 2.163 9.77 ]
        ci_high         [1.739 1.801 1.731 1.763 1.823 1.775 2.177 9.871]
        ci_comb_low     [0.3   0.307 0.316 0.321 0.324 0.32 ]
        ci_comb_high    [0.342 0.354 0.34  0.346 0.358 0.349]
        flags_comb      [False False False False False False]
        flags_indv      [False False False False False False False False]
        n_total         2^(12)
        n               [4096. 4096. 4096. 4096. 4096. 4096. 4096. 4096.]
        n_rep           [256. 256. 256. 256. 256. 256. 256. 256.]
        time_integrate  ...
    CubQMCCLT (StoppingCriterion Object)
        inflate         1.200
        alpha           0.010
        abs_tol         0.050
        rel_tol         0
        n_init          2^(8)
        n_max           2^(30)
        replications    2^(4)
    SobolIndices (Integrand Object)
        indices         [[0]
                        [1]
                        [2]]
    Gaussian (TrueMeasure Object)
        mean            0
        covariance      2^(-1)
        decomp_type     PCA
    DigitalNetB2 (DiscreteDistribution Object)
        d               6
        dvec            [0 1 2 3 4 5]
        randomize       LMS_DS
        graycode        0
        entropy         7
        spawn_key       (0,)
    
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
                Should not include [], the null set
        """
        self.parameters = ['indices']
        self.integrand = integrand
        self.d = self.integrand.d
        # indices
        if self.integrand.dprime>1: 
            raise ParameterError('SobolIndices currently only supports integrand.dprime = 1')
        self.indices = indices
        if self.indices=='singletons':
            self.indices = [[j] for j in range(self.d)]
        if [] in self.indices:
            raise ParameterError('SobolIndices indices cannot include [], the null set.')
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
        super(SobolIndices,self).__init__(parallel=False)
    
    def f(self, x, periodization_transform='NONE', *args, **kwargs):
        z = x[:,self.d:]
        x = x[:,:self.d]
        y = zeros((x.shape[0],self.dprime),dtype=float)
        compute_flags = kwargs['compute_flags']
        del kwargs['compute_flags']
        v = zeros(x.shape,dtype=float)
        f_x = self.integrand.f(x,periodization_transform,*args,**kwargs).squeeze()
        f_z = self.integrand.f(z,periodization_transform,*args,**kwargs).squeeze()
        for k in range(self.s):
            if not compute_flags[k]: continue
            u_bool = self.indices_bool_mat[k]
            not_u_bool = self.not_indices_bool_mat[k]
            v[:,u_bool] = x[:,u_bool]
            v[:,not_u_bool] = z[:,not_u_bool]
            f_v = self.integrand.f(v,periodization_transform,*args,**kwargs).squeeze()
            y[:,k] = f_x*(f_v-f_z) # A.18
            y[:,self.s+k] = (f_z-f_v)**2/2 # A.16
        if compute_flags[-2]: y[:,-2] = f_x # mu
        if compute_flags[-1]: y[:,-1] = f_x**2 # sigma^2+mu^2
        return y
    
    def _spawn(self, level, sampler):
        new_integrand = self.integrand.spawn(level,sampler)
        return SobolIndices(
            integrand = new_integrand,
            indices = self.indices)
    
    def bound_fun(self, bound_low, bound_high):
        f2_low,f2_high = bound_low[-1],bound_high[-1]
        mu_low,mu_high = bound_low[-2],bound_low[-2]
        sigma2_low,sigma2_high = f2_low-mu_high**2,f2_high-mu_low**2
        violated = sign(sigma2_low)!=sign(sigma2_high)
        bound_opts = [bound_low[:-2]/sigma2_low,bound_low[:-2]/sigma2_high,bound_high[:-2]/sigma2_low,bound_high[:-2]/sigma2_high]
        comb_bounds_low  = minimum.reduce(bound_opts)
        comb_bounds_high = maximum.reduce(bound_opts)
        return comb_bounds_low,comb_bounds_high,violated
    
    def dependency(self, flags_comb):
        numer_flags = flags_comb
        denom_flag = flags_comb.any()
        return hstack((numer_flags,denom_flag,denom_flag))
