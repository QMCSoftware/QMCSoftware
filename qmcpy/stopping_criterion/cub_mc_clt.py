from .abstract_stopping_criterion import AbstractStoppingCriterion
from ..util.data import Data

from ..discrete_distribution import IIDStdUniform
from ..discrete_distribution.abstract_discrete_distribution import AbstractIIDDiscreteDistribution
from ..true_measure import Gaussian, BrownianMotion, Uniform
from ..integrand import FinancialOption, Linear0, AbstractIntegrand
from ..util import MaxSamplesWarning,ParameterError
import numpy as np
from scipy.stats import norm
from time import time
import warnings


class CubMCCLT(AbstractStoppingCriterion):
    r"""
    IID Monte Carlo stopping criterion based on the Central Limit Theorem in a two step method.

    Examples:
        >>> ao = FinancialOption(IIDStdUniform(52,seed=7))
        >>> sc = CubMCCLT(ao,abs_tol=.05)
        >>> solution,data = sc.integrate()
        >>> data
        Data (Data)
            solution        1.777
            bound_low       1.723
            bound_high      1.831
            bound_diff      0.107
            n_total         74235
            time_integrate  ...
        CubMCCLT (AbstractStoppingCriterion)
            abs_tol         0.050
            rel_tol         0
            n_init          2^(10)
            n_limit         2^(30)
            inflate         1.200
            alpha           0.010
        FinancialOption (AbstractIntegrand)
            option          ASIAN
            call_put        CALL
            volatility      2^(-1)
            start_price     30
            strike_price    35
            interest_rate   0
            t_final         1
            asian_mean      ARITHMETIC
        BrownianMotion (AbstractTrueMeasure)
            time_vec        [0.019 0.038 0.058 ... 0.962 0.981 1.   ]
            drift           0
            mean            [0. 0. 0. ... 0. 0. 0.]
            covariance      [[0.019 0.019 0.019 ... 0.019 0.019 0.019]
                             [0.019 0.038 0.038 ... 0.038 0.038 0.038]
                             [0.019 0.038 0.058 ... 0.058 0.058 0.058]
                             ...
                             [0.019 0.038 0.058 ... 0.962 0.962 0.962]
                             [0.019 0.038 0.058 ... 0.962 0.981 0.981]
                             [0.019 0.038 0.058 ... 0.962 0.981 1.   ]]
            decomp_type     PCA
        IIDStdUniform (AbstractIIDDiscreteDistribution)
            d               52
            replications    1
            entropy         7
        
        Control variates 

        >>> iid = IIDStdUniform(52,seed=7)
        >>> ao = FinancialOption(iid,option="ASIAN")
        >>> eo = FinancialOption(iid,option="EUROPEAN")
        >>> lin0 = Linear0(iid)
        >>> sc = CubMCCLT(ao,abs_tol=.05,
        ...     control_variates = [eo,lin0],
        ...     control_variate_means = [eo.get_exact_value(),0])
        >>> solution,data = sc.integrate()
        >>> data
        Data (Data)
            solution        1.790
            bound_low       1.738
            bound_high      1.843
            bound_diff      0.104
            n_total         27777
            time_integrate  ...
        CubMCCLT (AbstractStoppingCriterion)
            abs_tol         0.050
            rel_tol         0
            n_init          2^(10)
            n_limit         2^(30)
            inflate         1.200
            alpha           0.010
            cv              [FinancialOption (AbstractIntegrand)
                                 option          EUROPEAN
                                 call_put        CALL
                                 volatility      2^(-1)
                                 start_price     30
                                 strike_price    35
                                 interest_rate   0
                                 t_final         1               Linear0 (AbstractIntegrand)]
            cv_mu           [4.211 0.   ]
        FinancialOption (AbstractIntegrand)
            option          ASIAN
            call_put        CALL
            volatility      2^(-1)
            start_price     30
            strike_price    35
            interest_rate   0
            t_final         1
            asian_mean      ARITHMETIC
        BrownianMotion (AbstractTrueMeasure)
            time_vec        [0.019 0.038 0.058 ... 0.962 0.981 1.   ]
            drift           0
            mean            [0. 0. 0. ... 0. 0. 0.]
            covariance      [[0.019 0.019 0.019 ... 0.019 0.019 0.019]
                             [0.019 0.038 0.038 ... 0.038 0.038 0.038]
                             [0.019 0.038 0.058 ... 0.058 0.058 0.058]
                             ...
                             [0.019 0.038 0.058 ... 0.962 0.962 0.962]
                             [0.019 0.038 0.058 ... 0.962 0.981 0.981]
                             [0.019 0.038 0.058 ... 0.962 0.981 1.   ]]
            decomp_type     PCA
        IIDStdUniform (AbstractIIDDiscreteDistribution)
            d               52
            replications    1
            entropy         7
    """

    def __init__(self, 
                 integrand, 
                 abs_tol = 1e-2, 
                 rel_tol = 0., 
                 n_init = 1024, 
                 n_limit = 2**30,
                 inflate = 1.2, 
                 alpha = 0.01, 
                 control_variates = [], 
                 control_variate_means = [],
                 ):
        r"""
        Args:
            integrand (AbstractIntegrand): The integrand.
            abs_tol (np.ndarray): Absolute error tolerance.
            rel_tol (np.ndarray): Relative error tolerance.
            n_init (int): Initial number of samples. 
            n_limit (int): Maximum number of samples.
            inflate (float): Inflation factor $\geq 1$ to multiply by the variance estimate to make it more conservative.
            alpha (np.ndarray): Uncertainty level in $(0,1)$. 
            control_variates (list): Integrands to use as control variates, each with the same underlying discrete distribution instance.
            control_variate_means (np.ndarray): Means of each control variate. 
        """
        self.parameters = ['abs_tol','rel_tol','n_init','n_limit','inflate','alpha']
        # Set Attributes
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.n_init = n_init
        self.n_limit = n_limit
        assert self.n_limit>(2*self.n_init), "require n_limit is at least twic as much as n_init"
        self.alpha = alpha
        self.inflate = inflate
        assert self.inflate>=1
        assert 0<self.alpha<1
        # QMCPy Objs
        self.integrand = integrand
        self.true_measure = self.integrand.true_measure
        self.discrete_distrib = self.integrand.discrete_distrib
        super(CubMCCLT,self).__init__(allowed_distribs=[AbstractIIDDiscreteDistribution],allow_vectorized_integrals=True)
        assert self.integrand.d_indv==()
        # control variates
        self.cv_mu = np.atleast_1d(control_variate_means)
        self.cv = control_variates
        if isinstance(self.cv,AbstractIntegrand):
            self.cv = [self.cv]
            self.cv_mu = self.cv_mu[None,...]
        assert isinstance(self.cv,list), "cv must be a list of AbstractIntegrand objects"
        for cv in self.cv:
            if (not isinstance(cv,AbstractIntegrand)) or (cv.discrete_distrib!=self.discrete_distrib) or (cv.d_indv!=self.integrand.d_indv):
                raise ParameterError('''
                        Each control variates discrete distribution must be an AbstractIntegrand instance 
                        with the same discrete distribution as the main integrand. d_indv must also match 
                        that of the main integrand instance for each control variate.''')
        self.ncv = len(self.cv)
        if self.ncv>0:
            assert self.cv_mu.shape==((self.ncv,)+self.integrand.d_indv), "Control variate means should have shape (len(control variates),d_indv)."
            self.parameters += ['cv','cv_mu']
        self.z_star = -norm.ppf(self.alpha/2.)

    def integrate(self):
        t_start = time()
        data = Data(
            parameters = [
                'solution',
                'bound_low',
                'bound_high',
                'bound_diff',
                'n_total',
                'time_integrate'])
        data.xfull = np.empty((0,self.integrand.d))
        data.yfull = np.empty(0)
        if self.ncv>0:
            data.ycvfull = np.empty((self.ncv,0))
        x0 = self.discrete_distrib(n=self.n_init)
        data.xfull = np.concatenate([data.xfull,x0],0)
        y0 = self.integrand.f(x0)
        temp_a = np.maximum(np.finfo(np.float64).eps,(time()-t_start)**0.5)
        data.yfull = np.concatenate([data.yfull,y0],-1)
        if self.ncv>0:
            ycv0 = [None]*self.ncv
            for k in range(self.ncv):
                ycv0[k] = self.cv[k].f(x0)
            ycv0 = np.stack(ycv0,0)
            data.ycvfull = np.concatenate([data.ycvfull,ycv0],1)
            cvmuhats = ycv0.mean(-1)
            x4beta = ycv0-cvmuhats[:,None]
            y4beta = y0-y0.mean()
            self.beta = np.linalg.lstsq(x4beta.T,y4beta,rcond=None)[0]
            y0 = y0-((ycv0-self.cv_mu[:,None])*self.beta[:,None]).sum(0)
        data.sighat0 = y0.std(ddof=1)
        data.solution0 = y0.mean()
        temp_b = temp_a*data.sighat0
        tol_up = np.maximum(self.abs_tol,abs(data.solution0)*self.rel_tol)
        n_mu_temp = np.ceil(temp_b*(data.sighat0/temp_a)*(self.z_star*self.inflate/tol_up)**2)
        data.n_mu = int(np.maximum(self.n_init,n_mu_temp))
        if (self.n_init+data.n_mu)>self.n_limit:
            # cannot generate this many new samples
            warning_s = """
            Already generated %d samples.
            Trying to generate %d new samples would exceed n_limit = %d.
            Will instead generate %d new samples to reach n_limit.""" \
            % (int(self.n_init),int(data.n_mu),int(self.n_limit),int(self.n_limit-self.n_init))
            warnings.warn(warning_s, MaxSamplesWarning)
            data.n_mu = self.n_limit-self.n_init
        x = self.discrete_distrib(n=data.n_mu)
        data.xfull = np.concatenate([data.xfull,x],0)
        y = self.integrand.f(x)
        data.yfull = np.concatenate([data.yfull,y],-1)
        if self.ncv>0:
            ycv = [None]*self.ncv
            for k in range(self.ncv):
                ycv[k] = self.cv[k].f(x)
            ycv = np.stack(ycv,0)
            data.ycvfull = np.concatenate([data.ycvfull,ycv],1)
            y = y-((ycv-self.cv_mu[:,None])*self.beta[:,None]).sum(0)
        data.sighat = y.std(ddof=1)
        data.solution = y.mean()
        data.bound_half_width = self.z_star*self.inflate*data.sighat/np.sqrt(data.n_mu)
        data.bound_low = data.solution-data.bound_half_width
        data.bound_high = data.solution+data.bound_half_width
        data.bound_diff = data.bound_high-data.bound_low
        data.n_total = self.n_init+data.n_mu 
        data.stopping_crit = self
        data.integrand = self.integrand
        data.true_measure = self.integrand.true_measure
        data.discrete_distrib = self.true_measure.discrete_distrib
        data.time_integrate = time()-t_start
        return data.solution,data

    def set_tolerance(self, abs_tol=None, rel_tol=None, rmse_tol=None):
        assert rmse_tol is None, "rmse_tol not supported by this stopping criterion."
        if abs_tol is not None:
            self.abs_tol = abs_tol
        if rel_tol is not None:
            self.rel_tol = rel_tol
