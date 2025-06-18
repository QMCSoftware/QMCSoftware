from ._cub_qmc_ml import _CubQMCML
from ..util.data import Data

from ..discrete_distribution import DigitalNetB2,Lattice,Halton
from ..discrete_distribution.abstract_discrete_distribution import AbstractLDDiscreteDistribution
from ..true_measure import Gaussian
from ..integrand import MLCallOptions
from ..util import MaxSamplesWarning, ParameterError, MaxLevelsWarning
import numpy as np
from scipy.stats import norm
from time import time
import warnings


class CubQMCML(_CubQMCML):
    """
    Stopping criterion based on multi-level quasi-Monte Carlo.

    >>> mlco = MLCallOptions(Lattice(seed=7,replications=32))
    >>> sc = CubQMCML(mlco,abs_tol=.075)
    >>> solution,data = sc.integrate()
    >>> data
    Data (Data)
        solution        10.418
        n_total         98304
        levels          5
        n_level         [2048  256  256  256  256]
        mean_level      [10.053  0.183  0.102  0.053  0.028]
        var_level       [2.454e-04 5.669e-05 2.597e-05 1.360e-05 7.235e-06]
        bias_estimate   0.016
        time_integrate  ...
    CubQMCML (AbstractStoppingCriterion)
        rmse_tol        0.029
        n_init          2^(8)
        n_max           10000000000
        replications    2^(5)
    MLCallOptions (AbstractIntegrand)
        option          european
        sigma           0.200
        k               100
        r               0.050
        t               1
        b               85
        level           0
    Gaussian (AbstractTrueMeasure)
        mean            0
        covariance      1
        decomp_type     PCA
    Lattice (AbstractLDDiscreteDistribution)
        d               1
        replications    2^(5)
        randomize       SHIFT
        gen_vec_source  kuo.lattice-33002-1024-1048576.9125.txt
        order           NATURAL
        n_limit         2^(20)
        entropy         7
    

    References:
        
        [1] M.B. Giles and B.J. Waterhouse. 'Multilevel quasi-Monte Carlo path simulation'.
        pp.165-181 in Advanced Financial Modelling, in Radon Series on Computational and Applied Mathematics,
        de Gruyter, 2009. http://people.maths.ox.ac.uk/~gilesm/files/radon.pdf
    """

    def __init__(self, integrand, abs_tol=.05, alpha=.01, rmse_tol=None, n_init=256, n_max=1e10, 
        levels_min=2, levels_max=10):
        """
        Args:
            integrand (AbstractIntegrand): integrand with multi-level g method
            abs_tol (float): absolute tolerance
            alpha (float): uncertainty level.
                If rmse_tol not supplied, then rmse_tol = abs_tol/norm.ppf(1-alpha/2)
            rmse_tol (float): root mean squared error
                If supplied (not None), then absolute tolerance and alpha are ignored
                in favor of the rmse tolerance
            n_max (int): maximum number of samples
            levels_min (int): minimum level of refinement >= 2
            levels_max (int): maximum level of refinement >= Lmin
        """
        self.parameters = ['rmse_tol','n_init','n_max','replications']
        # initialization
        if rmse_tol:
            self.rmse_tol = float(rmse_tol)
        else: # use absolute tolerance
            self.rmse_tol =  float(abs_tol) / norm.ppf(1-alpha/2)
        self.n_init = n_init
        self.n_max = n_max
        self.levels_min = levels_min
        self.levels_max = levels_max
        # QMCPy Objs
        self.integrand = integrand
        self.true_measure = self.integrand.true_measure
        self.discrete_distrib = self.true_measure.discrete_distrib
        super(CubQMCML,self).__init__(allowed_levels=['adaptive-multi'],allowed_distribs=[AbstractLDDiscreteDistribution],allow_vectorized_integrals=False)
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
            if (data.n_total + total_next_samples) > self.n_max:
                warning_s = """
                Already generated %d samples.
                Trying to generate %d new samples, which would exceed n_max = %d.
                Stopping integration process.
                Note that error tolerances may no longer be satisfied""" \
                % (int(data.n_total), int(total_next_samples), int(self.n_max))
                warnings.warn(warning_s, MaxSamplesWarning)
                break
        data.stopping_crit = self
        data.integrand = self.integrand
        data.true_measure = self.integrand.true_measure
        data.discrete_distrib = self.true_measure.discrete_distrib
        data.time_integrate = time()-t_start
        return data.solution,data
