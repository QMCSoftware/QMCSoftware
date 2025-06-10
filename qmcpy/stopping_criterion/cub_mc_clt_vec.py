from ._stopping_criterion import StoppingCriterion
from ..accumulate_data import MeanVarDataVec
from ..discrete_distribution._discrete_distribution import DiscreteDistribution
from ..discrete_distribution import IIDStdUniform
from ..discrete_distribution._discrete_distribution import IID
from ..true_measure import Gaussian,Uniform
from ..integrand import Keister,BoxIntegral,CustomFun
from ..util import MaxSamplesWarning, ParameterWarning, ParameterError
from numpy import *
from time import time
from scipy.stats import norm
import warnings


class CubMCCLTVec(StoppingCriterion):
    """
    Stopping criterion based on the Central Limit Theorem for vectorized integrands.

    >>> k = Keister(IIDStdUniform(seed=7))
    >>> sc = CubMCCLTVec(k,abs_tol=.05)
    >>> solution,data = sc.integrate()
    >>> abs(solution - 1.38366791) < 0.05
    array([ True])
    >>> f = BoxIntegral(IIDStdUniform(3,seed=7), s=[-1,1])
    >>> abs_tol = 5e-2
    >>> sc = CubMCCLTVec(f,abs_tol=abs_tol)
    >>> solution,data = sc.integrate()
    >>> import numpy as np
    >>> np.all(np.abs(solution - np.array([1.1853359 , 0.95670595])) < abs_tol)
    True
    >>> sol3neg1 = -pi/4-1/2*log(2)+log(5+3*sqrt(3))
    >>> sol31 = sqrt(3)/4+1/2*log(2+sqrt(3))-pi/24
    >>> true_value = array([sol3neg1,sol31])
    >>> np.all(np.abs(true_value-solution)<abs_tol)
    True
    >>> cf = CustomFun(
    ...     true_measure = Uniform(IIDStdUniform(6,seed=7)),
    ...     g = lambda x,compute_flags=None: (2*arange(1,7)*x).reshape(-1,2,3),
    ...     dimension_indv = (2,3))
    >>> sol,data = CubMCCLTVec(cf,abs_tol=1e-2).integrate()
    >>> np.all(np.abs(sol - np.array([[1., 1.999, 2.999],[4.001, 4.998, 6.001]])) < 1e-2)
    True
    """

    def __init__(self, integrand, abs_tol=1e-2, rel_tol=0., n_init=256., n_max=2**30,
        inflate=1.2, alpha=0.01,
        error_fun = lambda sv,abs_tol,rel_tol: maximum(abs_tol,abs(sv)*rel_tol)):
        """
        Args:
            integrand (Integrand): an instance of Integrand
            inflate (float): inflation factor when estimating variance
            alpha (ndarray): significance level for confidence interval
            abs_tol (ndarray): absolute error tolerance
            rel_tol (ndarray): relative error tolerance
            n_max (int): maximum number of samples
            error_fun: function taking in the approximate solution vector, 
                absolute tolerance, and relative tolerance which returns the approximate error. 
                Default indicates integration until either absolute OR relative tolerance is satisfied.
        """
        self.parameters = ['inflate','alpha','abs_tol','rel_tol','n_init','n_max']
        # Input Checks
        if log2(n_init) % 1 != 0:
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
        self.alphas_indv,identity_dependency = self._compute_indv_alphas(full(self.integrand.d_comb,self.alpha))
        self.z_star = -norm.ppf(self.alphas_indv/2)
         
    def integrate(self, resume=None):
        """ See abstract method. Optionally resumes from a previous computation.

        Args:
            resume (MeanVarDataVec, optional): Previous data object returned from a prior call to integrate. If provided, computation resumes from this state.
        """
        t_start = time()
        if resume is not None:
            self.data = resume
            # Optionally restore datum if present
            if hasattr(resume, 'datum'):
                self.datum = resume.datum
        else:
            self.datum = empty(self.d_indv,dtype=object)
            for j in ndindex(self.d_indv):
                self.datum[j] = MeanVarDataVec(self.z_star[j],self.inflate)
            self.data = MeanVarDataVec.__new__(MeanVarDataVec)
            self.data.flags_indv = tile(False,self.d_indv)
            self.data.compute_flags = tile(True,self.d_indv)
            self.data.indv_bound_low = tile(-inf,self.d_indv)
            self.data.indv_bound_high = tile(inf,self.d_indv)
            self.data.solution_indv = tile(nan,self.d_indv)
            self.data.solution = nan
            self.data.xfull = empty((0,self.d))
            self.data.yfull = empty((0,)+self.d_indv)
            n_min,n_max = 0,self.n_init
            self.data.n = tile(self.n_init,self.d_indv)
        while True:
            n = int(self.data.n.max())
            xnext = self.discrete_distrib.gen_samples(n)
            ynext = self.integrand.f(xnext,compute_flags=self.data.compute_flags)
            self.data.xfull = vstack((self.data.xfull,xnext))
            self.data.yfull = vstack((self.data.yfull,ynext))
            for j in ndindex(self.d_indv):
                if self.data.flags_indv[j]: continue
                yj = self.data.yfull[(slice(None),)+j]
                self.data.solution_indv[j],self.data.indv_bound_low[j],self.data.indv_bound_high[j] = self.datum[j].update_data(yj)
            self.data.comb_bound_low,self.data.comb_bound_high = self.integrand.bound_fun(self.data.indv_bound_low,self.data.indv_bound_high)
            self.abs_tols,self.rel_tols = full_like(self.data.comb_bound_low,self.abs_tol),full_like(self.data.comb_bound_low,self.rel_tol)
            fidxs = isfinite(self.data.comb_bound_low)&isfinite(self.data.comb_bound_high)
            slow,shigh,abs_tols,rel_tols = self.data.comb_bound_low[fidxs],self.data.comb_bound_high[fidxs],self.abs_tols[fidxs],self.rel_tols[fidxs]
            self.data.solution = tile(nan,self.data.comb_bound_low.shape)
            self.data.solution[fidxs] = 1/2*(slow+shigh+self.error_fun(slow,abs_tols,rel_tols)-self.error_fun(shigh,abs_tols,rel_tols))
            self.data.comb_flags = tile(False,self.data.comb_bound_low.shape)
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
        
        


    