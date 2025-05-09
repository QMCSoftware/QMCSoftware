from .abstract_stopping_criterion import AbstractStoppingCriterion
from ..accumulate_data import MeanVarDataRep
from ..discrete_distribution.abstract_discrete_distribution import AbstractDiscreteDistribution
from ..discrete_distribution import Lattice,DigitalNetB2,Halton
from ..discrete_distribution.abstract_discrete_distribution import AbstractLDDiscreteDistribution
from ..true_measure import Gaussian,Uniform
from ..integrand import Keister,BoxIntegral,CustomFun
from ..util import MaxSamplesWarning, NotYetImplemented, ParameterWarning, ParameterError
import numpy as np
from scipy.stats import t
from time import time
import warnings


class CubQMCCLT(AbstractStoppingCriterion):
    r"""
    Stopping criterion based on Student's $t$-distribution for multiple replications.
    
    Examples:
        >>> k = Keister(Lattice(seed=7,replications=25))
        >>> sc = CubQMCCLT(k,abs_tol=.05)
        >>> solution,data = sc.integrate()
        >>> solution
        array(1.38026356)
        >>> data
        MeanVarDataRep (AccumulateData)
            solution        1.380
            comb_bound_low  1.380
            comb_bound_high 1.381
            comb_bound_diff 0.001
            comb_flags      1
            n_total         6400
            n               6400
            n_rep           2^(8)
            time_integrate  ...
        CubQMCCLT (AbstractStoppingCriterion)
            inflate         1
            alpha           0.010
            abs_tol         0.050
            rel_tol         0
            n_init          2^(8)
            n_max           2^(30)
        Keister (AbstractIntegrand)
        Gaussian (AbstractTrueMeasure)
            mean            0
            covariance      2^(-1)
            decomp_type     PCA
        Lattice (AbstractLDDiscreteDistribution)
            d               1
            replications    25
            randomize       SHIFT
            gen_vec_source  kuo.lattice-33002-1024-1048576.9125.txt
            order           NATURAL
            n_limit         2^(20)
            entropy         7
        
        # >>> f = BoxIntegral(Lattice(3,seed=7), s=[-1,1])
        # >>> abs_tol = 1e-3
        # >>> sc = CubQMCCLT(f, abs_tol=abs_tol)
        # >>> solution,data = sc.integrate()
        # >>> solution
        # array([1.19029896, 0.96063374])
        # >>> data
        # MeanVarDataRep (AccumulateData Object)
        #     solution        [1.19  0.961]
        #     comb_bound_low  [1.19 0.96]
        #     comb_bound_high [1.191 0.961]
        #     comb_flags      [ True  True]
        #     n_total         2^(19)
        #     n               [524288.  16384.]
        #     n_rep           [32768.  1024.]
        #     time_integrate  ...
        # CubQMCCLT (AbstractStoppingCriterion Object)
        #     inflate         1.200
        #     alpha           0.010
        #     abs_tol         0.001
        #     rel_tol         0
        #     n_init          2^(8)
        #     n_max           2^(30)
        #     replications    2^(4)
        # BoxIntegral (AbstractIntegrand Object)
        #     s               [-1  1]
        # Uniform (AbstractTrueMeasure Object)
        #     lower_bound     0
        #     upper_bound     1
        # Lattice (AbstractDiscreteDistribution Object)
        #     d               3
        #     dvec            [0 1 2]
        #     randomize       SHIFT
        #     order           NATURAL
        #     gen_vec         [     1 182667 213731]
        #     entropy         7
        #     spawn_key       ()
        # >>> sol3neg1 = -np.pi/4-1/2*np.log(2)+np.log(5+3*np.sqrt(3))
        # >>> sol31 = np.sqrt(3)/4+1/2*np.log(2+np.sqrt(3))-np.pi/24
        # >>> true_value = np.array([sol3neg1,sol31])
        # >>> assert (abs(true_value-solution)<abs_tol).all()
        
        # >>> cf = CustomFun(
        # ...     true_measure = Uniform(DigitalNetB2(6,seed=7)),
        # ...     g = lambda x,compute_flags=None: (2*np.arange(1,7)*x).reshape(-1,2,3),
        # ...     dimension_indv = (2,3))
        # >>> sol,data = CubQMCCLT(cf,abs_tol=1e-4).integrate()
        # >>> data
        # MeanVarDataRep (AccumulateData Object)
        #     solution        [[1. 2. 3.]
        #                     [4. 5. 6.]]
        #     comb_bound_low  [[1. 2. 3.]
        #                     [4. 5. 6.]]
        #     comb_bound_high [[1. 2. 3.]
        #                     [4. 5. 6.]]
        #     comb_flags      [[ True  True  True]
        #                     [ True  True  True]]
        #     n_total         2^(13)
        #     n               [[4096. 4096. 4096.]
        #                     [8192. 4096. 4096.]]
        #     n_rep           [[256. 256. 256.]
        #                     [512. 256. 256.]]
        #     time_integrate  ...
        # CubQMCCLT (AbstractStoppingCriterion Object)
        #     inflate         1.200
        #     alpha           0.010
        #     abs_tol         1.00e-04
        #     rel_tol         0
        #     n_init          2^(8)
        #     n_max           2^(30)
        #     replications    2^(4)
        # CustomFun (AbstractIntegrand Object)
        # Uniform (AbstractTrueMeasure Object)
        #     lower_bound     0
        #     upper_bound     1
        # DigitalNetB2 (AbstractDiscreteDistribution Object)
        #     d               6
        #     dvec            [0 1 2 3 4 5]
        #     randomize       LMS_DS
        #     graycode        0
        #     entropy         7
        #     spawn_key       ()
    """

    def __init__(self,
                 integrand, 
                 abs_tol = 1e-2,
                 rel_tol = 0.,
                 n_init = 256.,
                 n_max = 2**30,
                 inflate = 1,
                 alpha = 0.01, 
                 error_fun = lambda sv,abs_tol,rel_tol: np.maximum(abs_tol,abs(sv)*rel_tol)):
        r"""
        Args:
            integrand (AbstractIntegrand): An AbstractIntegrand.
            inflate (float): Inflation factor to multiply by the variance estimate to make it more conservative. Must be greater than or equal to 1.
            alpha (np.ndarray): Uncertainty level in $(0,1)$. 
            abs_tol (np.ndarray): Absolute error tolerance.
            rel_tol (np.ndarray): Relative error tolerance.
            n_max (int): Maximum number of samples.
            error_fun (callable): Function mapping the approximate solution, absolute error tolerance, and relative error tolerance to the current error bound.

                - The default $(\hat{\boldsymbol{\mu}},\varepsilon_\mathrm{abs},\varepsilon_\mathrm{rel}) \mapsto \max\{\varepsilon_\mathrm{abs},\lvert \hat{\boldsymbol{\mu}} \rvert \varepsilon_\mathrm{rel}\}$ 
                    means the approximation error must be below either the absolue error *or* relative error.
                - Setting to $(\hat{\boldsymbol{\mu}},\varepsilon_\mathrm{abs},\varepsilon_\mathrm{rel}) \mapsto \min\{\varepsilon_\mathrm{abs},\lvert \hat{\boldsymbol{\mu}} \rvert \varepsilon_\mathrm{rel}\}$ 
                    means the approximation error must be below either the absolue error *and* relative error.  
        """
        self.parameters = ['inflate','alpha','abs_tol','rel_tol','n_init','n_max']
        # Input Checks
        if np.log2(n_init) % 1 != 0:
            warning_s = ' n_init must be a power of 2. Using n_init = 32'
            warnings.warn(warning_s, ParameterWarning)
            n_init = 32
        # Set Attributes
        self.n_init = float(n_init)
        self.n_max = float(n_max)
        self.alpha = alpha
        self.inflate = float(inflate)
        self.error_fun = error_fun
        # QMCPy Objs
        self.integrand = integrand
        super(CubQMCCLT,self).__init__(allowed_levels=["single"],allowed_distribs=[AbstractLDDiscreteDistribution],allow_vectorized_integrals=True)
        assert self.integrand.discrete_distrib.replications>1, "require the discrete distribution has replications>1"
        assert self.integrand.discrete_distrib.randomize!="FALSE", "Requires discrete distribution is randomized"
        self.alphas_indv,identity_dependency = self._compute_indv_alphas(np.full(self.integrand.d_comb,self.alpha))
        self.t_star = -t.ppf(self.alphas_indv/2,df=self.integrand.discrete_distrib.replications-1)
        self.set_tolerance(abs_tol,rel_tol)
        
    def integrate(self):
        t_start = time()
        self.data = MeanVarDataRep(self)
        while True:
            self.data.update_data()
            self.data.comb_bound_low,self.data.comb_bound_high = self.integrand.bound_fun(self.data.indv_bound_low,self.data.indv_bound_high)
            self.data.comb_bound_diff = self.data.comb_bound_high-self.data.comb_bound_low
            fidxs = np.isfinite(self.data.comb_bound_low)&np.isfinite(self.data.comb_bound_high)
            slow,shigh,abs_tols,rel_tols = self.data.comb_bound_low[fidxs],self.data.comb_bound_high[fidxs],self.abs_tols[fidxs],self.rel_tols[fidxs]
            self.data.solution = np.tile(np.nan,self.data.comb_bound_low.shape)
            self.data.solution[fidxs] = 1/2*(slow+shigh+self.error_fun(slow,abs_tols,rel_tols)-self.error_fun(shigh,abs_tols,rel_tols))
            self.data.comb_flags = np.tile(False,self.data.comb_bound_low.shape)
            self.data.comb_flags[fidxs] = (shigh-slow) <= (self.error_fun(slow,abs_tols,rel_tols)+self.error_fun(shigh,abs_tols,rel_tols))
            self.data.flags_indv = self.integrand.dependency(self.data.comb_flags)
            self.data.compute_flags = ~self.data.flags_indv
            if np.sum(self.data.compute_flags)==0:
                break # sufficiently estimated
            elif 2*self.data.n_total>self.n_max:
                warning_s = """
                Already generated %d samples.
                Trying to generate %d new samples would exceeds n_max = %d.
                No more samples will be generated.
                Note that error tolerances may not be satisfied. """ \
                % (int(self.data.n_total),int(self.data.n_total),int(self.n_max))
                warnings.warn(warning_s, MaxSamplesWarning)
                break
        self.data.time_integrate = time()-t_start
        return self.data.solution,self.data
    
    def set_tolerance(self, abs_tol=None, rel_tol=None):
        """
        See abstract method. 
        
        Args:
            abs_tol (float): absolute tolerance. Reset if supplied, ignored if not. 
            rel_tol (float): relative tolerance. Reset if supplied, ignored if not. 
        """
        if abs_tol is not None:
            self.abs_tol = abs_tol
            self.abs_tols = np.full(self.integrand.d_comb,self.abs_tol)
        if rel_tol is not None:
            self.rel_tol = rel_tol
            self.rel_tols = np.full(self.integrand.d_comb,self.rel_tol)

class CubQMCRep(CubQMCCLT): pass
