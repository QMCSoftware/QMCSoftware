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
    array([[0.32833328, 0.32747355, 0.32823724],
           [0.33871672, 0.33844826, 0.3387112 ]])
    >>> data
    LDTransformData (AccumulateData Object)
        solution        [[0.328 0.327 0.328]
                        [0.339 0.338 0.339]]
        indv_error      [[0.004 0.004 0.004 0.   ]
                        [0.002 0.002 0.002 0.003]]
        ci_low          [[1.671 1.667 1.671 2.168]
                        [1.725 1.724 1.725 9.797]]
        ci_high         [[1.679 1.674 1.678 2.169]
                        [1.73  1.729 1.73  9.803]]
        ci_comb_low     [[0.327 0.327 0.327]
                        [0.338 0.338 0.338]]
        ci_comb_high    [[0.329 0.328 0.329]
                        [0.339 0.339 0.339]]
        flags_comb      [[False False False]
                        [False False False]]
        flags_indv      [[False False False False]
                        [False False False False]]
        n_total         2^(15)
        n               [[32768. 32768. 32768. 32768.]
                        [32768. 32768. 32768. 32768.]]
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
    >>> cf = CustomFun(
    ...     true_measure = Uniform(dnb2),
    ...     g = lambda x,compute_flags=None: x,
    ...     dprime = 3)
    >>> keister_indices = SobolIndices(cf,indices='singletons')
    >>> sc = CubQMCNetG(keister_indices,abs_tol=1e-5)
    >>> solution,data = sc.integrate()
    >>> print(data)
    LDTransformData (AccumulateData Object)
        solution        [[[1. 0. 0.]
                         [0. 1. 0.]
                         [0. 0. 1.]]
    <BLANKLINE>
                        [[1. 0. 0.]
                         [0. 1. 0.]
                         [0. 0. 1.]]]
        indv_error      [[[7.396e-09 0.000e+00 0.000e+00]
                         [0.000e+00 2.340e-07 0.000e+00]
                         [0.000e+00 0.000e+00 1.551e-07]
                         [4.547e-12 0.000e+00 0.000e+00]]
    <BLANKLINE>
                        [[5.584e-09 0.000e+00 0.000e+00]
                         [0.000e+00 2.073e-07 0.000e+00]
                         [0.000e+00 0.000e+00 1.178e-08]
                         [1.708e-09 3.107e-08 4.336e-08]]]
        ci_low          [[[0.083 0.    0.   ]
                         [0.    0.083 0.   ]
                         [0.    0.    0.083]
                         [0.5   0.5   0.5  ]]
    <BLANKLINE>
                        [[0.083 0.    0.   ]
                         [0.    0.083 0.   ]
                         [0.    0.    0.083]
                         [0.333 0.333 0.333]]]
        ci_high         [[[0.083 0.    0.   ]
                         [0.    0.083 0.   ]
                         [0.    0.    0.083]
                         [0.5   0.5   0.5  ]]
    <BLANKLINE>
                        [[0.083 0.    0.   ]
                         [0.    0.083 0.   ]
                         [0.    0.    0.083]
                         [0.333 0.333 0.333]]]
        ci_comb_low     [[[1. 0. 0.]
                         [0. 1. 0.]
                         [0. 0. 1.]]
    <BLANKLINE>
                        [[1. 0. 0.]
                         [0. 1. 0.]
                         [0. 0. 1.]]]
        ci_comb_high    [[[1. 0. 0.]
                         [0. 1. 0.]
                         [0. 0. 1.]]
    <BLANKLINE>
                        [[1. 0. 0.]
                         [0. 1. 0.]
                         [0. 0. 1.]]]
        flags_comb      [[[False False False]
                         [False False False]
                         [False False False]]
    <BLANKLINE>
                        [[False False False]
                         [False False False]
                         [False False False]]]
        flags_indv      [[[False False False]
                         [False False False]
                         [False False False]
                         [False False False]]
    <BLANKLINE>
                        [[False False False]
                         [False False False]
                         [False False False]
                         [False False False]]]
        n_total         2^(15)
        n               [[[32768.  1024.  1024.]
                         [ 1024. 16384.  1024.]
                         [ 1024.  1024. 16384.]
                         [32768. 16384. 32768.]]
    <BLANKLINE>
                        [[32768.  1024.  1024.]
                         [ 1024. 16384.  1024.]
                         [ 1024.  1024. 32768.]
                         [32768. 16384. 32768.]]]
        time_integrate  ...
    CubQMCNetG (StoppingCriterion Object)
        abs_tol         1.00e-05
        rel_tol         0
        n_init          2^(10)
        n_max           2^(35)
    SobolIndices (Integrand Object)
        indices         [[0]
                        [1]
                        [2]]
    Uniform (TrueMeasure Object)
        lower_bound     0
        upper_bound     1
    DigitalNetB2 (DiscreteDistribution Object)
        d               6
        dvec            [0 1 2 3 4 5]
        randomize       LMS_DS
        graycode        0
        entropy         7
        spawn_key       (1,)
    
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
        self.true_measure = self.integrand.true_measure
        self.discrete_distrib = self.true_measure.discrete_distrib.spawn(s=1,dimensions=[self.dtilde])[0]
        self.sampler = self.integrand.sampler
        dprime = (2,self.s+1,)+self.integrand.dprime
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
            if not flags_total.any(): continue
            u_bool = self.indices_bool_mat[k]
            not_u_bool = self.not_indices_bool_mat[k]
            v[:,u_bool] = x[:,u_bool]
            v[:,not_u_bool] = z[:,not_u_bool]
            f_v = self.integrand.f(v,compute_flags=flags_k,*args,**kwargs)
            y[:,0,k] = f_x*(f_v-f_z) # A.18
            y[:,1,k] = (f_z-f_v)**2/2 # A.16
        y[:,0,-1] = f_x # mu
        y[:,1,-1] = f_x**2 # sigma^2+mu^2
        return y
    
    def _spawn(self, level, sampler):
        new_integrand = self.integrand.spawn(level,sampler)
        return SobolIndices(
            integrand = new_integrand,
            indices = self.indices)
    
    def bound_fun(self, bound_low, bound_high):
        mu_low,mu_high = bound_low[0,-1],bound_low[0,-1]
        f2_low,f2_high = bound_low[1,-1],bound_high[1,-1]
        sigma2_low,sigma2_high = f2_low-mu_high**2,f2_high-mu_low**2
        violated = sign(sigma2_low)!=sign(sigma2_high)
        bl_sl = bound_low[:,:-1]/sigma2_low
        bl_sh = bound_low[:,:-1]/sigma2_high
        bh_sl = bound_high[:,:-1]/sigma2_low
        bh_sh = bound_high[:,:-1]/sigma2_high
        comb_bounds_low  = minimum.reduce([bl_sl,bl_sh,bh_sl,bh_sh])
        comb_bounds_high = maximum.reduce([bl_sl,bl_sh,bh_sl,bh_sh])
        return comb_bounds_low,comb_bounds_high,violated
    
    def dependency(self, flags_comb):
        individual_flags = zeros(self.dprime,dtype=bool)
        individual_flags[:,:-1] = flags_comb # numerator
        closed_flags = flags_comb[0]
        total_flags = flags_comb[1]
        individual_flags[:,-1] = (closed_flags|total_flags).any(0)
        return individual_flags
