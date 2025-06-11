from ._stopping_criterion import StoppingCriterion
from ..accumulate_data import MeanVarData
from ..discrete_distribution import IIDStdUniform
from ..discrete_distribution._discrete_distribution import IID
from ..true_measure import Gaussian, BrownianMotion, Uniform
from ..integrand import Keister, AsianOption, CustomFun
from ..util import MaxSamplesWarning
from numpy import *
from scipy.stats import norm
from time import time
import warnings


class CubMCCLT(StoppingCriterion):
    """
    Stopping criterion based on the Central Limit Theorem.
    
    >>> ao = AsianOption(IIDStdUniform(seed=7))
    >>> sc = CubMCCLT(ao,abs_tol=.05)
    >>> solution,data = sc.integrate()
    >>> data
    MeanVarData (AccumulateData Object)
        solution        1.519
        error_bound     0.046
        n_total         96028
        n               95004
        levels          1
        time_integrate  ...
    CubMCCLT (StoppingCriterion Object)
        abs_tol         0.050
        rel_tol         0
        n_init          2^(10)
        n_max           10000000000
        inflate         1.200
        alpha           0.010
    AsianOption (Integrand Object)
        volatility      2^(-1)
        call_put        call
        start_price     30
        strike_price    35
        interest_rate   0
        mean_type       arithmetic
        dim_frac        0
    BrownianMotion (TrueMeasure Object)
        time_vec        1
        drift           0
        mean            0
        covariance      1
        decomp_type     PCA
    IIDStdUniform (DiscreteDistribution Object)
        d               1
        entropy         7
        spawn_key       ()
    >>> ao = AsianOption(IIDStdUniform(seed=7),multilevel_dims=[2,4,8])
    >>> sc = CubMCCLT(ao,abs_tol=.05)
    >>> solution,data = sc.integrate()
    >>> dd = IIDStdUniform(1,seed=7)
    >>> k = Keister(dd)
    >>> cv1 = CustomFun(Uniform(dd),lambda x: sin(pi*x).sum(1))
    >>> cv1mean = 2/pi
    >>> cv2 = CustomFun(Uniform(dd),lambda x: (-3*(x-.5)**2+1).sum(1))
    >>> cv2mean = 3/4
    >>> sc1 = CubMCCLT(k,abs_tol=.05,control_variates=[cv1,cv2],control_variate_means=[cv1mean,cv2mean])
    >>> sol,data = sc1.integrate()
    >>> data
    MeanVarData (AccumulateData Object)
        solution        1.381
        error_bound     0.010
        n_total         3072
        n               2^(11)
        levels          1
        time_integrate  ...
    CubMCCLT (StoppingCriterion Object)
        abs_tol         0.050
        rel_tol         0
        n_init          2^(10)
        n_max           10000000000
        inflate         1.200
        alpha           0.010
    Keister (Integrand Object)
    Gaussian (TrueMeasure Object)
        mean            0
        covariance      2^(-1)
        decomp_type     PCA
    IIDStdUniform (DiscreteDistribution Object)
        d               1
        entropy         7
        spawn_key       ()
    """

    def __init__(self, integrand, abs_tol=1e-2, rel_tol=0., n_init=1024., n_max=1e10,
        inflate=1.2, alpha=0.01, control_variates=[], control_variate_means=[],
        error_fun = lambda sv,abs_tol,rel_tol: maximum(abs_tol,abs(sv)*rel_tol)):
        """
        Args:
            integrand (Integrand): an instance of Integrand
            inflate (float): inflation factor when estimating variance
            alpha (float): significance level for confidence interval
            abs_tol (ndarray): absolute error tolerance
            rel_tol (ndarray): relative error tolerance
            n_max (int): maximum number of samples
            control_variates (list): list of integrand objects to be used as control variates. 
                Control variates are currently only compatible with single level problems. 
                The same discrete distribution instance must be used for the integrand and each of the control variates. 
            control_variate_means (list): list of means for each control variate
        """
        self.parameters = ['abs_tol','rel_tol','n_init','n_max','inflate','alpha']
        # Set Attributes
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.n_init = n_init
        self.n_max = n_max
        self.alpha = alpha
        self.inflate = inflate
        # QMCPy Objs
        self.integrand = integrand
        self.true_measure = self.integrand.true_measure
        self.discrete_distrib = self.integrand.discrete_distrib
        self.cv = control_variates
        self.cv_mu = control_variate_means
        # Verify Compliant Construction
        allowed_levels = ['single','fixed-multi']
        allowed_distribs = [IID]
        allow_vectorized_integrals = True
        super(CubMCCLT,self).__init__(allowed_levels, allowed_distribs, allow_vectorized_integrals)

    def integrate(self, resume=None):
        """ See abstract method. Optionally resumes from a previous computation.
        
        Args:
            resume (MeanVarData, optional): Previous data object returned from a prior call to integrate. 
                If provided, computation resumes from this state.
        """
        t_start = time()
        if resume is not None:
            self.data = resume
        else:
            # Construct AccumulateData Object to House Integration data
            self.data = MeanVarData(self, self.integrand, self.true_measure, self.discrete_distrib, 
                self.n_init, self.cv, self.cv_mu)  # house integration data
        # Pilot Sample
        self.data.update_data()
        # use cost of function values to decide how to allocate
        temp_a = self.data.t_eval ** 0.5
        temp_b = (temp_a * self.data.sighat).sum()
        # samples for computation of the mean
        # n_mu_temp := n such that confidence intervals width and confidence will be satisfied
        tol_up = max(self.abs_tol, abs(self.data.solution) * self.rel_tol)
        z_star = -norm.ppf(self.alpha / 2.)
        n_mu_temp = ceil(temp_b * (self.data.sighat / temp_a) * (z_star * self.inflate / tol_up)**2)
        # n_mu := n_mu_temp adjusted for previous n
        self.data.n_mu = maximum(self.data.n, n_mu_temp)
        self.data.n += self.data.n_mu.astype(int)
        if self.data.n_total + self.data.n.sum() > self.n_max:
            # cannot generate this many new samples
            warning_s = """
            Already generated %d samples.
            Trying to generate %d new samples, which would exceed n_max = %d.
            The number of new samples will be decrease proportionally for each integrand.
            Note that error tolerances may no longer be satisfied.""" \
            % (int(self.data.n_total), int(self.data.n.sum()), int(self.n_max))
            warnings.warn(warning_s, MaxSamplesWarning)
            # decrease n proportionally for each integrand
            n_decease = self.data.n_total + self.data.n.sum() - self.n_max
            dec_prop = n_decease / self.data.n.sum()
            self.data.n = floor(self.data.n - self.data.n * dec_prop)
        # Final Sample
        self.data.update_data()
        # CLT confidence interval
        sigma_up = (self.data.sighat ** 2 / self.data.n_mu).sum(0) ** 0.5
        self.data.error_bound = z_star * self.inflate * sigma_up
        self.data.confid_int = self.data.solution + self.data.error_bound * array([-1, 1])
        self.data.time_integrate = time() - t_start
        return self.data.solution, self.data

    def set_tolerance(self, abs_tol=None, rel_tol=None):
        """
        See abstract method. 
        
        Args:
            abs_tol (float): absolute tolerance. Reset if supplied, ignored if not. 
            rel_tol (float): relative tolerance. Reset if supplied, ignored if not. 
        """
        if abs_tol != None: self.abs_tol = abs_tol
        if rel_tol != None: self.rel_tol = rel_tol