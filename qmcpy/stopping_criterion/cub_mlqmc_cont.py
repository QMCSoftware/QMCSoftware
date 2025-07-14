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


class CubMLQMCCont(AbstractCubMLQMC):
    """
    Multilevel Quasi-Monte Carlo stopping criterion with continuation.

    Examples:
        >>> fo = FinancialOption(DigitalNetB2(seed=7,replications=32))
        >>> sc = CubMLQMCCont(fo,abs_tol=1e-3)
        >>> solution,data = sc.integrate()
        >>> data
        Data (Data)
            solution        1.784
            n_total         4718592
            levels          2^(2)
            n_level         [65536 32768 32768 16384]
            mean_level      [1.718 0.051 0.012 0.003]
            var_level       [6.589e-09 2.091e-08 1.701e-08 7.554e-08]
            bias_estimate   2.55e-04
            time_integrate  ...
        CubMLQMCCont (AbstractStoppingCriterion)
            rmse_tol        3.88e-04
            n_init          2^(8)
            n_limit         10000000000
            replications    2^(5)
            levels_min      2^(1)
            levels_max      10
            n_tols          10
            inflate         1.668
            theta_init      2^(-1)
            theta           2^(-3)
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
        
    1.  [https://github.com/PieterjanRobbe/MultilevelEstimators.jl](https://github.com/PieterjanRobbe/MultilevelEstimators.jl).
    """

    def __init__(self, 
                 integrand, 
                 abs_tol = .05, 
                 rmse_tol = None,
                 n_init = 256,
                 n_limit = 1e10,
                 inflate = 100**(1/9),
                 alpha = .01,   
                 levels_min = 2, 
                 levels_max = 10, 
                 n_tols = 10, 
                 theta_init = 0.5,
                 ):
        r"""
        Args:
            integrand (AbstractIntegrand): The integrand.
            abs_tol (np.ndarray): Absolute error tolerance.
            rmse_tol (np.ndarray): Root mean squared error tolerance. 
                If supplied, then absolute tolerance and alpha are ignored in favor of the rmse tolerance. 
            n_init (int): Initial number of samples. 
            n_limit (int): Maximum number of samples.
            inflate (float): Coarser tolerance multiplication factor $\geq 1$.
            alpha (np.ndarray): Uncertainty level in $(0,1)$. 
            levels_min (int): Minimum level of refinement $\geq 2$.
            levels_max (int): Maximum level of refinement $\geq$ `levels_min`.
            n_tols (int): Number of coarser tolerances to run.
            theta_init (float): Initial error splitting constant.
        """
        self.parameters = ['rmse_tol','n_init','n_limit','replications','levels_min',
            'levels_max','n_tols','inflate','theta_init','theta']
        # initialization
        if rmse_tol:
            self.target_tol = float(rmse_tol)
        else: # use absolute tolerance
            self.target_tol =  float(abs_tol) / norm.ppf(1-alpha/2)
        self.n_init = n_init
        self.n_limit = n_limit
        self.levels_min = levels_min
        self.levels_max = levels_max
        self.theta_init = theta_init
        self.theta = theta_init
        self.n_tols = n_tols
        self.alpha = alpha
        self.inflate = inflate
        assert self.inflate>=1
        assert 0<self.alpha<1
        # QMCPy Objs
        self.integrand = integrand
        self.true_measure = self.integrand.true_measure
        self.discrete_distrib = self.integrand.discrete_distrib
        super(CubMLQMCCont,self).__init__(allowed_distribs=[AbstractLDDiscreteDistribution],allow_vectorized_integrals=False)
        self.replications = self.discrete_distrib.replications 
        assert self.replications>=4, "require at least 4 replications"

    def integrate(self):
        t_start = time()
        data = self._construct_data()
        # Loop over coarser tolerances
        for t in range(self.n_tols):
            self.rmse_tol = self.inflate**(self.n_tols-t-1)*self.target_tol # Set new target tolerance
            self._integrate(data)
        data.stopping_crit = self
        data.integrand = self.integrand
        data.true_measure = self.integrand.true_measure
        data.discrete_distrib = self.true_measure.discrete_distrib
        data.time_integrate = time()-t_start
        return data.solution,data

    def _construct_data(self):
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
        return data 
    
    def _integrate(self, data):
        #self.theta = self.theta_init
        data.levels = int(self.levels_min+1)

        converged = False
        while not converged:
            # Ensure that we have samples on the finest level
            self.update_data(data)
            self._update_theta(data)

            while self._varest(data) > (1-self.theta)*self.rmse_tol**2:
                efficient_level = np.argmax(data.var_cost_ratio_level[:data.levels])
                data.eval_level[efficient_level] = True

                # Check if over sample budget
                total_next_samples = (self.replications*data.eval_level*data.n_level*2).sum()
                if (data.n_total + total_next_samples) > self.n_limit:
                    warning_s = """
                    Already generated %d samples.
                    Trying to generate %d new samples, which would exceed n_limit = %d.
                    Stopping integration process.
                    Note that error tolerances may no longer be satisfied""" \
                    % (int(data.n_total), int(total_next_samples), int(self.n_limit))
                    warnings.warn(warning_s, MaxSamplesWarning)
                    return

                self.update_data(data)
                self._update_theta(data)

            # Check for convergence
            converged = self._rmse(data) < self.rmse_tol
            if not converged:
                if data.levels == self.levels_max:
                    warnings.warn(
                        'Failed to achieve weak convergence. levels == levels_max.',
                        MaxLevelsWarning)
                    converged = True
                else:
                    self._add_level(data)

    def _update_theta(self, data):
        # Update error splitting parameter
        max_levels = len(data.n_level)
        A = np.ones((2,2))
        A[:,0] = range(max_levels-2, max_levels)
        y = np.ones(2)
        y[0] = np.log2(abs(data.mean_level_reps[max_levels-2].mean()))
        y[1] = np.log2(abs(data.mean_level_reps[max_levels-1].mean()))
        x = np.linalg.lstsq(A, y, rcond=None)[0]
        alpha = max(.5,-x[0])
        real_bias = 2**(x[1]+max_levels*x[0]) / (2**alpha - 1)
        self.theta = max(0.01, min(0.125, (real_bias/self.rmse_tol)**2))

    def _rmse(self, data):
        # Returns an estimate for the root mean square error
        return np.sqrt(self._mse(data))

    def _mse(self, data):
        # Returns an estimate for the mean square error
        return (1-self.theta)*self._varest(data) + self.theta*data.bias_estimate**2

    def _varest(self, data):
        # Returns the variance of the estimator
        return data.var_level[:data.levels].sum()
