from ._cub_mc_ml import _CubMCML
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


class CubMCML(_CubMCML):
    """
    Stopping criterion based on multi-level monte carlo.
    
    Examples:
        >>> fo = FinancialOption(IIDStdUniform(seed=7))
        >>> sc = CubMCML(fo,abs_tol=1.5e-2)
        >>> solution,data = sc.integrate()
        >>> data
        Data (Data)
            solution        1.785
            n_total         3577556
            levels          2^(2)
            n_level         [2438191  490331  207606   62905]
            mean_level      [1.715 0.053 0.013 0.003]
            var_level       [21.829  1.761  0.452  0.11 ]
            cost_per_sample [ 2.  4.  8. 16.]
            alpha           2.008
            beta            1.997
            gamma           1.000
            time_integrate  ...
        CubMCML (AbstractStoppingCriterion)
            rmse_tol        0.006
            n_init          2^(8)
            levels_min      2^(1)
            levels_max      10
            theta           2^(-1)
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

    Original Implementation:

        http://people.maths.ox.ac.uk/~gilesm/mlmc/#MATLAB

    References:
        
        [1] M.B. Giles. 'Multi-level Monte Carlo path simulation'. 
        Operations Research, 56(3):607-617, 2008.
        http://people.maths.ox.ac.uk/~gilesm/files/OPRE_2008.pdf.
    """

    def __init__(self, integrand, abs_tol=.05, alpha=.01, rmse_tol=None, n_init=256, n_max=1e10, 
        levels_min=2, levels_max=10, alpha0=-1., beta0=-1., gamma0=-1.):
        """
        Args:
            integrand (AbstractIntegrand): integrand with multi-level g method
            abs_tol (float): absolute tolerance. Reset if supplied, ignored if not. 
            alpha (float): uncertainty level.
                If rmse_tol not supplied, then rmse_tol = abs_tol/norm.ppf(1-alpha/2)
            rel_tol (float): relative tolerance. Reset if supplied, ignored if not.
                Takes priority over absolute tolerance and alpha if supplied. 
            n_init (int): initial number of samples
            n_max (int): maximum number of samples
            levels_min (int): minimum level of refinement >= 2
            levels_max (int): maximum level of refinement >= Lmin
            alpha0 (float): weak error is O(2^{-alpha0*level})
            beta0 (float): variance is O(2^{-bet0a*level})
            gamma0 (float): sample cost is O(2^{gamma0*level})
        
        Note:
            if alpha, beta, gamma are not positive, then they will be estimated
        """
        self.parameters = ['rmse_tol','n_init','levels_min','levels_max','theta']
        if levels_min < 2:
            raise ParameterError('needs levels_min >= 2')
        if levels_max < levels_min:
            raise ParameterError('needs levels_max >= levels_min')
        if n_init <= 0:
            raise ParameterError('needs n_init > 0')
        # initialization
        if rmse_tol:
            self.rmse_tol = float(rmse_tol)
        else: # use absolute tolerance
            self.rmse_tol =  float(abs_tol) / norm.ppf(1-alpha/2)
        self.n_init = n_init
        self.n_max = n_max
        self.levels_min = levels_min
        self.levels_max = levels_max
        self.theta = 0.5
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.gamma0 = gamma0
        # QMCPy Objs
        self.integrand = integrand
        self.true_measure = self.integrand.true_measure
        self.discrete_distrib = self.true_measure.discrete_distrib
        super(CubMCML,self).__init__(allowed_levels=['adaptive-multi'],allowed_distribs=[AbstractIIDDiscreteDistribution], allow_vectorized_integrals=False)

    def integrate(self):
        t_start = time()
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
        while data.diff_n_level.sum() > 0:
            self._update_data(data)
            data.n_total += data.diff_n_level.sum()
            # set optimal number of additional samples
            n_samples = self._get_next_samples(data)
            data.diff_n_level = np.maximum(0, n_samples-data.n_level)
            # check if over sample budget
            if (data.n_total + data.diff_n_level.sum()) > self.n_max:
                warning_s = """
                Already generated %d samples.
                Trying to generate %d new samples, which would exceed n_max = %d.
                Stopping integration process.
                Note that error tolerances may no longer be satisfied""" \
                % (int(data.n_total), int(data.diff_n_level.sum()), int(self.n_max))
                warnings.warn(warning_s, MaxSamplesWarning)
                break
            # if (almost) converged, estimate remaining error and decide 
            # whether a new level is required
            if (data.diff_n_level > 0.01*data.n_level).sum() == 0:
                range_ = np.arange(min(2.,data.levels-1)+1)
                idx = (data.levels-range_).astype(int)
                rem = (data.mean_level[idx] / 
                        2.**(range_*data.alpha)).max() / (2.**data.alpha - 1)
                # rem = ml(l+1) / (2^alpha - 1)
                if rem > np.sqrt(self.theta)*self.rmse_tol:
                    if data.levels == self.levels_max:
                        warnings.warn(
                            'Failed to achieve weak convergence. levels == levels_max.',
                            MaxLevelsWarning)
                    else:
                        self._add_level(data)
                        n_samples = self._get_next_samples(data)
                        data.diff_n_level = np.maximum(0,n_samples-data.n_level)
        # finally, evaluate multilevel estimator
        data.solution = (data.sum_level[0,:]/data.n_level).sum()
        data.levels += 1
        data.stopping_crit = self
        data.integrand = self.integrand
        data.true_measure = self.integrand.true_measure
        data.discrete_distrib = self.true_measure.discrete_distrib
        data.time_integrate = time()-t_start
        return data.solution,data
