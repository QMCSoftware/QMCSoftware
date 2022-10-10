from ._integrand import Integrand
from . import Keister, BoxIntegral
from ..stopping_criterion import CubQMCNetG
from ..util import ParameterError
from ..true_measure import Uniform
from ..discrete_distribution import DigitalNetB2
from numpy import *
from itertools import combinations


class SensitivityIndices(Integrand):
    """
    Sensitivity' Indicies, normalized Sobol' Indices. 

    >>> dnb2 = DigitalNetB2(dimension=3,seed=7)
    >>> keister_d = Keister(dnb2)
    >>> keister_indices = SensitivityIndices(keister_d,indices='singletons')
    >>> sc = CubQMCNetG(keister_indices,abs_tol=1e-3)
    >>> solution,data = sc.integrate()
    >>> solution.squeeze()
    array([[0.32803639, 0.32795358, 0.32807359],
           [0.33884667, 0.33857811, 0.33884115]])
    >>> data
    LDTransformData (AccumulateData Object)
        solution        [[0.328 0.328 0.328]
                        [0.339 0.339 0.339]]
        comb_bound_low  [[0.327 0.327 0.328]
                        [0.338 0.338 0.338]]
        comb_bound_high [[0.329 0.329 0.329]
                        [0.34  0.339 0.34 ]]
        comb_flags      [[ True  True  True]
                        [ True  True  True]]
        n_total         2^(16)
        n               [[[65536. 65536. 65536.]
                         [65536. 65536. 65536.]
                         [65536. 65536. 65536.]]
    <BLANKLINE>
                        [[32768. 32768. 32768.]
                         [32768. 32768. 32768.]
                         [32768. 32768. 32768.]]]
        time_integrate  ...
    CubQMCNetG (StoppingCriterion Object)
        abs_tol         0.001
        rel_tol         0
        n_init          2^(10)
        n_max           2^(35)
    SensitivityIndices (Integrand Object)
        indices         [[0]
                        [1]
                        [2]]
        n_multiplier    3
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
    >>> sc = CubQMCNetG(SobolIndices(BoxIntegral(DigitalNetB2(3,seed=7)),indices='all'),abs_tol=.01)
    >>> sol,data = sc.integrate()
    >>> print(sol)
    [[[0.32312991 0.33340559]
      [0.32331463 0.33342669]
      [0.32160276 0.33318619]
      [0.65559598 0.6667154 ]
      [0.65551702 0.66670251]
      [0.6556618  0.66672429]]
    <BLANKLINE>
     [[0.3440018  0.33341845]
      [0.34501082 0.33347005]
      [0.34504829 0.33345212]
      [0.67659368 0.6667021 ]
      [0.67725088 0.66667925]
      [0.67802866 0.66672587]]]
    
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
                Should not include [], the null set, or [0,...,d-1], the set of all indices. 
                Setting indices='all' will compute all sensitivity indices
        """
        self.parameters = ['indices','n_multiplier']
        self.integrand = integrand
        self.d = self.integrand.d
        # indices
        self.indices = indices
        if self.indices=='singletons':
            self.indices = [[j] for j in range(self.d)]
        elif self.indices=='all':
            self.indices = []
            for r in range(1,self.d):
                self.indices += [list(idx) for idx in combinations(arange(self.d),r)]
        if [] in self.indices or [i for i in range(self.d)] in self.indices:
            raise ParameterError('SensitivityIndices indices cannot include [], the null set.')
        self.s = len(self.indices)
        self.indices_bool_mat = tile(False,(self.s,self.d))
        for k in range(self.s): self.indices_bool_mat[k,self.indices[k]] = True
        self.not_indices_bool_mat = ~self.indices_bool_mat
        # sensitivity_index
        self.n_multiplier = self.s
        self.dtilde = 2*self.d
        self.true_measure = self.integrand.true_measure
        self.discrete_distrib = self.true_measure.discrete_distrib.spawn(s=1,dimensions=[self.dtilde])[0]
        self.sampler = self.integrand.sampler
        super(SensitivityIndices,self).__init__(
            rho = (2,3,self.s,)+self.integrand.rho,
            eta = (2,self.s,)+self.integrand.rho,
            parallel = False)
    
    def f(self, x, *args, **kwargs):
        z = x[:,self.d:]
        x = x[:,:self.d]
        n,d = x.shape
        y = zeros((n,)+self.rho,dtype=float)
        compute_flags = kwargs['compute_flags']
        del kwargs['compute_flags']
        v = zeros((n,d),dtype=float)
        f_x = self.integrand.f(x,*args,**kwargs)
        f_z = self.integrand.f(z,*args,**kwargs)
        for k in range(self.s):
            flags_closed = compute_flags[0,0,k,:]
            flags_total = compute_flags[1,0,k,:]
            flags_k = flags_closed|flags_total
            if not flags_k.any(): continue
            u_bool = self.indices_bool_mat[k]
            not_u_bool = self.not_indices_bool_mat[k]
            v[:,u_bool] = x[:,u_bool]
            v[:,not_u_bool] = z[:,not_u_bool]
            f_v = self.integrand.f(v,compute_flags=flags_k,*args,**kwargs)
            y[:,0,0,k] = f_x*(f_v-f_z) # A.18
            y[:,1,0,k] = (f_z-f_v)**2/2 # A.16
            y[:,:,1,k] = f_x[:,None,:] # mu
            y[:,:,2,k] = f_x[:,None,:]**2 # sigma^2+mu^2
        return y
    
    def _spawn(self, level, sampler):
        new_integrand = self.integrand.spawn(level,sampler)
        return SensitivityIndices(
            integrand = new_integrand,
            indices = self.indices)
    
    def bound_fun(self, bound_low, bound_high):
        tau_low,mu_low,f2_low = bound_low[:,0],bound_low[:,1],bound_low[:,2]
        tau_high,mu_high,f2_high = bound_high[:,0],bound_high[:,1],bound_high[:,2]
        sigma2_low1,sigma2_low2 = f2_high-mu_low**2,f2_high-mu_high**2
        comb_bounds_low = clip(minimum.reduce([tau_low/sigma2_low1,tau_low/sigma2_low2]),0,1)
        sigma2_high1,sigma2_high2 = f2_low-mu_low**2,f2_low-mu_high**2
        comb_bounds_high = clip(maximum.reduce([tau_high/sigma2_high1,tau_high/sigma2_high2]),0,1)
        violated = (sigma2_high1<=0)|(sigma2_high2<=0)
        comb_bounds_low[violated],comb_bounds_high[violated] = 0,1
        return comb_bounds_low,comb_bounds_high
    
    def dependency(self, comb_flags):
        return repeat(comb_flags[:,None,:],3,axis=1)

class SobolIndices(SensitivityIndices):
    """ Normalized Sobol' Indices, an alias for SensitivityIndices. """
    pass