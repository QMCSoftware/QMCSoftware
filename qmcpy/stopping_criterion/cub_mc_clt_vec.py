from ._stopping_criterion import StoppingCriterion
from ..accumulate_data import MeanVarDataVec
from ..discrete_distribution._discrete_distribution import DiscreteDistribution
from ..discrete_distribution import IIDStdUniform
from ..discrete_distribution._discrete_distribution import IID
from ..true_measure import Gaussian,Uniform
from ..integrand import Keister,BoxIntegral,CustomFun
from ..util import MaxSamplesWarning, ParameterWarning, ParameterError
import numpy as np
from time import time
from scipy.stats import norm
import warnings


class CubMCCLTVec(StoppingCriterion):
    """
    Stopping criterion based on the Central Limit Theorem for vectorized integrands.
    
    >>> k = Keister(IIDStdUniform(seed=7))
    >>> sc = CubMCCLTVec(k,abs_tol=.05)
    >>> solution,data = sc.integrate()
    >>> solution
    np.array([1.38798425])
    >>> data
    MeanVarDataVec (AccumulateData Object)
        solution        1.388
        comb_bound_low  1.343
        comb_bound_high 1.433
        comb_flags      1
        n_total         2^(10)
        n               2^(10)
        time_integrate  ...
    CubMCCLTVec (StoppingCriterion Object)
        inflate         1.200
        alpha           0.010
        abs_tol         0.050
        rel_tol         0
        n_init          2^(8)
        n_max           2^(30)
    Keister (Integrand Object)
    Gaussian (TrueMeasure Object)
        mean            0
        covariance      2^(-1)
        decomp_type     PCA
    IIDStdUniform (DiscreteDistribution Object)
        d               1
        entropy         7
        spawn_key       ()
    >>> f = BoxIntegral(IIDStdUniform(3,seed=7), s=[-1,1])
    >>> abs_tol = 5e-2
    >>> sc = CubMCCLTVec(f,abs_tol=abs_tol)
    >>> solution,data = sc.integrate()
    >>> solution
    np.array([1.16031586, 0.96023843])
    >>> data
    MeanVarDataVec (AccumulateData Object)
        solution        [1.16 0.96]
        comb_bound_low  [1.112 0.923]
        comb_bound_high [1.209 0.997]
        comb_flags      [ True  True]
        n_total         2^(10)
        n               [1024.  512.]
        time_integrate  ...
    CubMCCLTVec (StoppingCriterion Object)
        inflate         1.200
        alpha           0.010
        abs_tol         0.050
        rel_tol         0
        n_init          2^(8)
        n_max           2^(30)
    BoxIntegral (Integrand Object)
        s               [-1  1]
    Uniform (TrueMeasure Object)
        lower_bound     0
        upper_bound     1
    IIDStdUniform (DiscreteDistribution Object)
        d               3
        entropy         7
        spawn_key       ()
    >>> sol3neg1 = -pi/4-1/2*np.log(2)+np.log(5+3*np.sqrt(3))
    >>> sol31 = np.sqrt(3)/4+1/2*np.log(2+np.sqrt(3))-pi/24
    >>> true_value = np.array([sol3neg1,sol31])
    >>> assert (abs(true_value-solution)<abs_tol).all()
    >>> cf = CustomFun(
    ...     true_measure = Uniform(IIDStdUniform(6,seed=7)),
    ...     g = lambda x,compute_flags=None: (2*np.arange(1,7)*x).reshape(-1,2,3),
    ...     dimension_indv = (2,3))
    >>> sol,data = CubMCCLTVec(cf,abs_tol=1e-2).integrate()
    >>> data
    MeanVarDataVec (AccumulateData Object)
        solution        [[1.    1.996 3.   ]
                        [4.001 5.003 5.999]]
        comb_bound_low  [[0.99  1.986 2.993]
                        [3.991 4.994 5.991]]
        comb_bound_high [[1.01  2.006 3.007]
                        [4.011 5.012 6.006]]
        comb_flags      [[ True  True  True]
                        [ True  True  True]]
        n_total         2^(21)
        n               [[  32768.  131072.  524288.]
                        [ 524288. 1048576. 2097152.]]
        time_integrate  ...
    CubMCCLTVec (StoppingCriterion Object)
        inflate         1.200
        alpha           0.010
        abs_tol         0.010
        rel_tol         0
        n_init          2^(8)
        n_max           2^(30)
    CustomFun (Integrand Object)
    Uniform (TrueMeasure Object)
        lower_bound     0
        upper_bound     1
    IIDStdUniform (DiscreteDistribution Object)
        d               6
        entropy         7
        spawn_key       ()
    """

    def __init__(self, integrand, abs_tol=1e-2, rel_tol=0., n_init=256., n_max=2**30,
        inflate=1.2, alpha=0.01,
        error_fun = lambda sv,abs_tol,rel_tol: maximum(abs_tol,abs(sv)*rel_tol)):
        """
        Args:
            integrand (Integrand): an instance of Integrand
            inflate (float): inflation factor when estimating variance
            alpha (np.ndarray): significance level for confidence interval
            abs_tol (np.ndarray): absolute error tolerance
            rel_tol (np.ndarray): relative error tolerance
            n_max (int): maximum number of samples
            error_fun: function taking in the approximate solution vector, 
                absolute tolerance, and relative tolerance which returns the approximate error. 
                Default indicates integration until either absolute OR relative tolerance is satisfied.
        """
        self.parameters = ['inflate','alpha','abs_tol','rel_tol','n_init','n_max']
        # Input Checks
        if np.log2(n_init) % 1 != 0:
            warning_s = ' n_init must be a power of 2. Using n_init = 32'
            warnings.warn(warning_s, ParameterWarning)
            n_init = 32
        # Set Attributes
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.n_init = float(n_init)
        self.n_max = float(n_max)
        self.alpha = alpha
        self.inflate = float(inflate)
        self.error_fun = error_fun
        # QMCPy Objs
        self.integrand = integrand
        self.true_measure = self.integrand.true_measure
        self.discrete_distrib = self.integrand.discrete_distrib
        self.d_indv = self.integrand.d_indv
        self.d = self.discrete_distrib.d
        super(CubMCCLTVec,self).__init__(allowed_levels=["single"], allowed_distribs=[IID], allow_vectorized_integrals=True)
        self.alphas_indv,identity_dependency = self._compute_indv_alphas(np.full(self.integrand.d_comb,self.alpha))
        self.z_star = -norm.ppf(self.alphas_indv/2)
         
    def integrate(self):
        """ See abstract method. """
        t_start = time()
        self.datum = np.empty(self.d_indv,dtype=object)
        for j in ndindex(self.d_indv):
            self.datum[j] = MeanVarDataVec(self.z_star[j],self.inflate)
        self.data = MeanVarDataVec.__new__(MeanVarDataVec)
        self.data.flags_indv = np.tile(False,self.d_indv)
        self.data.compute_flags = np.tile(True,self.d_indv)
        self.data.indv_bound_low = np.tile(-np.inf,self.d_indv)
        self.data.indv_bound_high = np.tile(np.inf,self.d_indv)
        self.data.solution_indv = np.tile(nan,self.d_indv)
        self.data.solution = nan
        self.data.xfull = np.empty((0,self.d))
        self.data.yfull = np.empty((0,)+self.d_indv)
        n_min,n_max = 0,self.n_init
        self.data.n = np.tile(self.n_init,self.d_indv)
        while True:
            n = int(n_max-n_min)
            xnext = self.discrete_distrib.gen_samples(n)
            ynext = self.integrand.f(xnext,compute_flags=self.data.compute_flags)
            self.data.xfull = np.vstack((self.data.xfull,xnext))
            self.data.yfull = np.vstack((self.data.yfull,ynext))
            for j in ndindex(self.d_indv):
                if self.data.flags_indv[j]: continue
                yj = self.data.yfull[(slice(None),)+j]
                self.data.solution_indv[j],self.data.indv_bound_low[j],self.data.indv_bound_high[j] = self.datum[j].update_data(yj)
            self.data.comb_bound_low,self.data.comb_bound_high = self.integrand.bound_fun(self.data.indv_bound_low,self.data.indv_bound_high)
            self.abs_tols,self.rel_tols = full_like(self.data.comb_bound_low,self.abs_tol),full_like(self.data.comb_bound_low,self.rel_tol)
            fidxs = isfinite(self.data.comb_bound_low)&isfinite(self.data.comb_bound_high)
            slow,shigh,abs_tols,rel_tols = self.data.comb_bound_low[fidxs],self.data.comb_bound_high[fidxs],self.abs_tols[fidxs],self.rel_tols[fidxs]
            self.data.solution = np.tile(nan,self.data.comb_bound_low.shape)
            self.data.solution[fidxs] = 1/2*(slow+shigh+self.error_fun(slow,abs_tols,rel_tols)-self.error_fun(shigh,abs_tols,rel_tols))
            self.data.comb_flags = np.tile(False,self.data.comb_bound_low.shape)
            self.data.comb_flags[fidxs] = (shigh-slow) <= (self.error_fun(slow,abs_tols,rel_tols)+self.error_fun(shigh,abs_tols,rel_tols))
            self.data.flags_indv = self.integrand.dependency(self.data.comb_flags)
            self.data.compute_flags = ~self.data.flags_indv
            self.data.n_total = self.data.n.max()
            if sum(self.data.compute_flags)==0:
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
            else:
                n_min = n_max; n_max = 2*n_min
                self.data.n += self.data.n*(self.data.compute_flags) # double sample size
        self.data.integrand = self.integrand
        self.data.true_measure = self.true_measure
        self.data.discrete_distrib = self.discrete_distrib
        self.data.stopping_crit = self
        self.data.parameters = [
            'solution',
            'comb_bound_low',
            'comb_bound_high',
            'comb_flags',
            'n_total',
            'n',
            'time_integrate']
        self.data.datum = self.datum
        self.data.time_integrate = time()-t_start
        return self.data.solution,self.data
    
    def set_tolerance(self, abs_tol=None, rel_tol=None):
        """
        See abstract method. 
        
        Args:
            abs_tol (float): absolute tolerance. Reset if supplied, ignored if not. 
            rel_tol (float): relative tolerance. Reset if supplied, ignored if not. 
        """
        if abs_tol != None: self.abs_tol = abs_tol
        if rel_tol != None: self.rel_tol = rel_tol
        
        


    
