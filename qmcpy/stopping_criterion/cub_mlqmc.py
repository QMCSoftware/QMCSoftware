from .abstract_cub_mlqmc import AbstractCubMLQMC
from ..util.data import Data
from ..discrete_distribution import DigitalNetB2,Lattice,Halton
from ..discrete_distribution.abstract_discrete_distribution import AbstractLDDiscreteDistribution
from ..true_measure import Gaussian
from ..integrand import FinancialOption
from ..util import MaxSamplesWarning, ParameterError, MaxLevelsWarning
import numpy as np
from scipy.stats import norm
from time import time
import warnings


class CubMLQMC(AbstractCubMLQMC):
    """
    Multilevel Quasi-Monte Carlo stopping criterion.

    Examples:
        >>> fo = FinancialOption(DigitalNetB2(seed=7,replications=32))
        >>> sc = CubMLQMC(fo,abs_tol=3e-3)
        >>> solution,data = sc.integrate()
        >>> data
        Data (Data)
            solution        1.784
            n_total         2097152
            levels          2^(2)
            n_level         [32768 16384  8192  8192]
            mean_level      [1.718 0.051 0.012 0.003]
            var_level       [4.607e-08 5.753e-08 3.305e-07 1.557e-07]
            bias_estimate   2.66e-04
            time_integrate  ...
        CubMLQMC (AbstractStoppingCriterion)
            rmse_tol        0.001
            n_init          2^(8)
            n_limit         10000000000
            replications    2^(5)
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
            time_vec        1
            drift           0
            mean            0
            covariance      1
            decomp_type     PCA
        DigitalNetB2 (AbstractLDDiscreteDistribution)
            d               1
            replications    2^(5)
            randomize       LMS DS
            gen_mats_source joe_kuo.6.21201.txt
            order           RADICAL INVERSE
            t               63
            alpha           1
            n_limit         2^(32)
            entropy         7

    **References:**
        
    1.  M.B. Giles and B.J. Waterhouse.  
        'Multilevel quasi-Monte Carlo path simulation'.  
        pp.165-181 in Advanced Financial Modelling, in Radon Series on Computational and Applied Mathematics, de Gruyter, 2009.  
        [http://people.maths.ox.ac.uk/~gilesm/files/radon.pdf](http://people.maths.ox.ac.uk/~gilesm/files/radon.pdf).
    """

    def __init__(self, 
                 integrand, 
                 abs_tol = .05, 
                 rmse_tol = None, 
                 n_init = 256, 
                 n_limit = 1e10,
                 alpha = .01, 
                 levels_min = 2,
                 levels_max = 10,
                 ):
        r"""
        Args:
            integrand (AbstractIntegrand): The integrand.
            abs_tol (np.ndarray): Absolute error tolerance.
            rmse_tol (np.ndarray): Root mean squared error tolerance. 
                If supplied, then absolute tolerance and alpha are ignored in favor of the rmse tolerance. 
            n_init (int): Initial number of samples. 
            n_limit (int): Maximum number of samples.
            alpha (np.ndarray): Uncertainty level in $(0,1)$. 
            levels_min (int): Minimum level of refinement $\geq 2$.
            levels_max (int): Maximum level of refinement $\geq$ `levels_min`.
        """
        self.parameters = ['rmse_tol','n_init','n_limit','replications']
        # initialization
        if rmse_tol:
            self.rmse_tol = float(rmse_tol)
        else: # use absolute tolerance
            self.rmse_tol =  float(abs_tol) / norm.ppf(1-alpha/2)
        self.alpha = alpha 
        assert 0<self.alpha<1
        self.n_init = n_init
        self.n_limit = n_limit
        self.levels_min = levels_min
        self.levels_max = levels_max
        # QMCPy Objs
        self.integrand = integrand
        self.true_measure = self.integrand.true_measure
        self.discrete_distrib = self.integrand.discrete_distrib
        super(CubMLQMC,self).__init__(allowed_distribs=[AbstractLDDiscreteDistribution],allow_vectorized_integrals=False)
        self.replications = self.discrete_distrib.replications 
        assert self.replications>=4, "require at least 4 replications"

    def integrate(self):
        t_start = time()
        data = Data(parameters=[
            'solution',
            'n_total',
            'levels',
            'n_level',
            'mean_level',
            'var_level', 
            'bias_estimate'])
        data.levels = int(self.levels_min+1)
        data.n_level = np.zeros(data.levels,dtype=int)
        data.eval_level = np.ones(data.levels,dtype=bool)
        data.mean_level_reps = np.zeros((data.levels,int(self.replications)))
        data.mean_level = np.tile(0.,data.levels)
        data.var_level = np.tile(np.inf,data.levels)
        data.cost_level = np.tile(0.,data.levels)
        data.var_cost_ratio_level = np.tile(np.inf,data.levels)
        data.bias_estimate = np.inf
        data.level_integrands = []
        while True:
            self.update_data(data)
            if data.var_level.sum() > (self.rmse_tol**2/2.):
                # double N_l on level with largest V_l/(2^l*N_l)
                efficient_level = np.argmax(data.var_cost_ratio_level)
                data.eval_level[efficient_level] = True
            elif data.bias_estimate > (self.rmse_tol/np.sqrt(2.)):
                if data.levels == self.levels_max + 1:
                        warnings.warn("Failed to achieve weak convergence. levels == levels_max.", MaxLevelsWarning)
                # add another level
                self._add_level(data)
            else:
                # both conditions met
                break
            total_next_samples = (self.replications*data.eval_level*data.n_level*2).sum()
            if (data.n_total + total_next_samples) > self.n_limit:
                warning_s = """
                Already generated %d samples.
                Trying to generate %d new samples, which would exceed n_limit = %d.
                Stopping integration process.
                Note that error tolerances may no longer be satisfied""" \
                % (int(data.n_total), int(total_next_samples), int(self.n_limit))
                warnings.warn(warning_s, MaxSamplesWarning)
                break
        data.stopping_crit = self
        data.integrand = self.integrand
        data.true_measure = self.integrand.true_measure
        data.discrete_distrib = self.true_measure.discrete_distrib
        data.time_integrate = time()-t_start
        return data.solution,data
