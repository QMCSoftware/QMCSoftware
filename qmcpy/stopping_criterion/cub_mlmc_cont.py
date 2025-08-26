from .abstract_stopping_criterion import AbstractStoppingCriterion
from .abstract_cub_mlmc import AbstractCubMLMC
from ..util.data import Data

from ..discrete_distribution import IIDStdUniform
from ..discrete_distribution.abstract_discrete_distribution import AbstractIIDDiscreteDistribution
from ..true_measure import Gaussian
from ..integrand import FinancialOption
from ..util import MaxSamplesWarning, ParameterError, MaxLevelsWarning, ParameterWarning
import numpy as np
from scipy.stats import norm
from time import time
import warnings


class CubMLMCCont(AbstractCubMLMC):
    r"""
    Multilevel IID Monte Carlo stopping criterion with continuation.
    
    Examples:
        >>> fo = FinancialOption(IIDStdUniform(seed=7))
        >>> sc = CubMLMCCont(fo,abs_tol=1.5e-2)
        >>> solution,data = sc.integrate()
        >>> data
        Data (Data)
            solution        1.771
            n_total         2291120
            levels          3
            n_level         [1094715  222428   79666     912     256]
            mean_level      [1.71  0.048 0.012]
            var_level       [21.826  1.768  0.453]
            cost_per_sample [2. 4. 8.]
            alpha           1.970
            beta            1.965
            gamma           1.000
            time_integrate  ...
        CubMLMCCont (AbstractStoppingCriterion)
            rmse_tol        0.006
            n_init          2^(8)
            levels_min      2^(1)
            levels_max      10
            n_tols          10
            inflate         1.668
            theta_init      2^(-1)
            theta           0.010
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
        IIDStdUniform (AbstractIIDDiscreteDistribution)
            d               1
            replications    1
            entropy         7

    **References:**

    1. [https://github.com/PieterjanRobbe/MultilevelEstimators.jl](https://github.com/PieterjanRobbe/MultilevelEstimators.jl).
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
        self.parameters = ['rmse_tol','n_init','levels_min','levels_max','n_tols',
            'inflate','theta_init','theta']
        if levels_min < 2:
            raise ParameterError('needs levels_min >= 2')
        if levels_max < levels_min:
            raise ParameterError('needs levels_max >= levels_min')
        if n_init <= 0:
            raise ParameterError('needs n_init > 0')
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
        self.inflate = inflate
        self.integrand = integrand
        self.true_measure = self.integrand.true_measure
        self.discrete_distrib = self.integrand.discrete_distrib
        self.alpha0 = -1 
        self.beta0 = -1 
        self.gamma0 = -1
        self.alpha = alpha
        self.inflate = inflate
        assert self.inflate>=1
        assert 0<self.alpha<1
        super(CubMLMCCont,self).__init__(allowed_distribs=[AbstractIIDDiscreteDistribution],allow_vectorized_integrals=False)

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
            'cost_per_sample',
            'alpha',
            'beta',
            'gamma'])
        data.levels = int(self.levels_min)
        data.n_level = np.zeros(data.levels+1,dtype=int)
        data.sum_level = np.zeros((2,data.levels+1))
        data.cost_level = np.zeros(data.levels+1)
        data.diff_n_level = self.n_init*np.ones(data.levels+1,dtype=int)
        data.alpha = np.maximum(0,self.alpha0)
        data.beta = np.maximum(0,self.beta0)
        data.gamma = np.maximum(0,self.gamma0)
        data.level_integrands = []
        return data

    def _integrate(self, data):
        self.theta = self.theta_init
        data.levels = int(self.levels_min)
        self._update_data(data) # Take warm-up samples if none have been taken so far

        converged = False
        while not converged:

            # Check if we already have samples at the finest level
            if not data.n_level[data.levels] > 0:
                # This takes n_init warm-up samples at the finest level
                data.diff_n_level = np.hstack((data.diff_n_level,self.n_init))
                self._update_data(data)
                # Alternatively, evaluate optimal number of samples and take between 2 and n_init samples
                #data.diff_n_level = self._get_next_samples(data)
                #data.diff_n_level[:data.levels] = 0
                #data.diff_n_level[data.levels] = max(3, min(self.n_init, data.diff_n_level[data.levels]))

            # Update splitting parameter
            self._update_theta(data)

            # Set optimal number of additional samples
            n_samples = self._get_next_samples(data)
            data.diff_n_level = np.maximum(0, n_samples-data.n_level[:data.levels+1])
            
            # Check if over sample budget
            if (data.n_total + data.diff_n_level.sum()) > self.n_limit:
                warning_s = """
                Already generated %d samples.
                Trying to generate %d new samples, which would exceed n_limit = %d.
                Stopping integration process.
                Note that error tolerances may no longer be satisfied""" \
                % (int(data.n_total), int(data.diff_n_level.sum()), int(self.n_limit))
                warnings.warn(warning_s, MaxSamplesWarning)
                break

            # Take additional samples
            self._update_data(data)
            data.n_total += data.diff_n_level.sum()

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

        data.diff_n_level.fill(0)
        data.solution = (data.sum_level[0,:data.levels+1]/data.n_level[:data.levels+1]).sum()
        data.levels += 1

    def _update_theta(self, data):
        # Update error splitting parameter
        self.theta = max(0.01, min(0.5, (self._bias(data,len(data.n_level)-1)/self.rmse_tol)**2))

    def _rmse(self, data):
        # Returns an estimate for the root mean square error
        return np.sqrt(self._mse(data))

    def _mse(self, data):
        # Returns an estimate for the mean square error
        return (1-self.theta)*self._varest(data) + self.theta*self._bias(data,data.levels)**2

    def _varest(self, data):
        # Returns the variance of the estimator
        return (data.var_level/data.n_level[:data.levels+1]).sum()

    def _bias(self, data, level):
        # Returns an estimate for the bias
        mean_level = data.sum_level[0, :]/data.n_level
        A = np.ones((2,2))
        A[:,0] = range(level-1, level+1)
        y = np.log2(abs(mean_level[level-1:level+1]))
        x = np.linalg.lstsq(A, y, rcond=None)[0]
        alpha = max(.5,-x[0])
        return 2**(x[1]+(level+1)*x[0]) / (2**alpha - 1)
