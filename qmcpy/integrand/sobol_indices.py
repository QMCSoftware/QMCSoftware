from ._integrand import Integrand
from . import Keister, CustomFun
from ..stopping_criterion import CubQMCNetG
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
    >>> sc = CubQMCNetG(keister_indices,abs_tol=1e-3)
    >>> solution,data = sc.integrate()
    >>> solution.squeeze()
    array([[0.32803639, 0.32795358, 0.32807359],
           [0.33884667, 0.33857811, 0.33884115]])
    >>> data
    LDTransformData (AccumulateData Object)
        solution        [[0.328 0.328 0.328]
                        [0.339 0.339 0.339]]
        indv_error      [[0.002 0.002 0.002]
                        [0.002 0.002 0.002]
                        [0.    0.    0.   ]
                        [0.    0.    0.   ]
                        [0.001 0.001 0.001]
                        [0.003 0.003 0.003]]
        ci_low          [[1.67  1.67  1.671]
                        [1.725 1.724 1.725]
                        [2.168 2.168 2.168]
                        [2.168 2.168 2.168]
                        [9.799 9.799 9.799]
                        [9.797 9.797 9.797]]
        ci_high         [[1.675 1.674 1.675]
                        [1.73  1.729 1.73 ]
                        [2.168 2.168 2.168]
                        [2.169 2.169 2.169]
                        [9.802 9.802 9.802]
                        [9.803 9.803 9.803]]
        ci_comb_low     [[0.327 0.327 0.328]
                        [0.338 0.338 0.338]]
        ci_comb_high    [[0.329 0.329 0.329]
                        [0.34  0.339 0.34 ]]
        flags_comb      [[False False False]
                        [False False False]]
        flags_indv      [[False False False]
                        [False False False]
                        [False False False]
                        [False False False]
                        [False False False]
                        [False False False]]
        n_total         2^(16)
        n               [[65536. 65536. 65536.]
                        [32768. 32768. 32768.]
                        [65536. 65536. 65536.]
                        [32768. 32768. 32768.]
                        [65536. 65536. 65536.]
                        [32768. 32768. 32768.]]
        time_integrate  ...
    CubQMCNetG (StoppingCriterion Object)
        abs_tol         0.001
        rel_tol         0
        n_init          2^(10)
        n_max           2^(35)
    SobolIndices (Integrand Object)
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
        self.parameters = ['indices','n_multiplier']
        self.integrand = integrand
        self.d = self.integrand.d
        # indices
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
        self.n_multiplier = self.s
        self.dtilde = 2*self.d
        self.true_measure = self.integrand.true_measure
        self.discrete_distrib = self.true_measure.discrete_distrib.spawn(s=1,dimensions=[self.dtilde])[0]
        self.sampler = self.integrand.sampler
        dprime = (6,self.s,)+self.integrand.dprime
        super(SobolIndices,self).__init__(dprime,parallel=False)
    
    def f(self, x, *args, **kwargs):
        z = x[:,self.d:]
        x = x[:,:self.d]
        n,d = x.shape
        y = zeros((n,)+self.dprime,dtype=float)
        compute_flags = kwargs['compute_flags']
        del kwargs['compute_flags']
        v = zeros((n,d),dtype=float)
        f_x = self.integrand.f(x,*args,**kwargs)
        f_z = self.integrand.f(z,*args,**kwargs)
        for k in range(self.s):
            flags_closed = compute_flags[0,k,:]
            flags_total = compute_flags[1,k,:]
            flags_k = flags_closed|flags_total
            if not flags_k.any(): continue
            u_bool = self.indices_bool_mat[k]
            not_u_bool = self.not_indices_bool_mat[k]
            v[:,u_bool] = x[:,u_bool]
            v[:,not_u_bool] = z[:,not_u_bool]
            f_v = self.integrand.f(v,compute_flags=flags_k,*args,**kwargs)
            y[:,0,k] = f_x*(f_v-f_z) # A.18
            y[:,1,k] = (f_z-f_v)**2/2 # A.16
            y[:,2,k] = f_x # mu
            y[:,3,k] = f_x # mu copy
            y[:,4,k] = f_x**2 # sigma^2+mu^2
            y[:,5,k] = f_x**2 # sigma^2+mu^2 copy
        return y
    
    def _spawn(self, level, sampler):
        new_integrand = self.integrand.spawn(level,sampler)
        return SobolIndices(
            integrand = new_integrand,
            indices = self.indices)
    
    def bound_fun(self, bound_low, bound_high):
        tau_low,mu_low,f2_low = bound_low[:2],bound_low[2:4],bound_low[4:6]
        tau_high,mu_high,f2_high = bound_high[:2],bound_high[2:4],bound_high[4:6]
        sigma2_low1,sigma2_low2 = f2_high-mu_low**2,f2_high-mu_high**2
        comb_bounds_low = minimum.reduce([tau_low/sigma2_low1,tau_low/sigma2_low2])
        comb_bounds_low = minimum.reduce([ones(tau_low.shape),maximum.reduce([zeros(tau_low.shape),comb_bounds_low])])
        sigma2_high1,sigma2_high2 = f2_low-mu_low**2,f2_low-mu_high**2
        comb_bounds_high = minimum.reduce([ones(tau_high.shape),maximum.reduce([tau_high/sigma2_high1,tau_high/sigma2_high2])])
        comb_bounds_high = minimum.reduce([ones(tau_high.shape),maximum.reduce([zeros(tau_high.shape),comb_bounds_high])])
        violated = (sigma2_high1<0)|(sigma2_high2<0)
        return comb_bounds_low,comb_bounds_high,violated
    
    def dependency(self, flags_comb):
        individual_flags = zeros(self.dprime,dtype=bool)
        individual_flags[:2] = flags_comb # numerator
        individual_flags[2:4] = flags_comb # copy to mu flags
        individual_flags[4:6] = flags_comb # copy to second moment flags
        return individual_flags

class SensitivityIndices(SobolIndices): pass