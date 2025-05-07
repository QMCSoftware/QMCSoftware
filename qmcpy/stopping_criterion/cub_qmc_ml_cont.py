from ._stopping_criterion import StoppingCriterion
from ..accumulate_data import MLQMCData
from ..discrete_distribution import DigitalNetB2,Lattice,Halton
from ..discrete_distribution.abstract_discrete_distribution import AbstractLDDiscreteDistribution
from ..true_measure import Gaussian
from ..integrand import MLCallOptions
from ..util import MaxSamplesWarning, ParameterError, MaxLevelsWarning
import numpy as np
from scipy.stats import norm
from time import time
import warnings


class CubQMCMLCont(StoppingCriterion):
    """
    Stopping criterion based on continuation multi-level quasi-Monte Carlo.

    >>> mlco = MLCallOptions(Lattice(seed=7))
    >>> sc = CubQMCMLCont(mlco,abs_tol=.05)
    >>> solution,data = sc.integrate()
    >>> data
    MLQMCData (AccumulateData Object)
        solution        10.420
        n_total         98304
        n_level         [2048.  256.  256.  256.  256.]
        levels          5
        mean_level      [10.054  0.183  0.102  0.054  0.027]
        var_level       [2.027e-04 5.129e-05 3.243e-05 1.610e-05 5.633e-06]
        bias_estimate   0.014
        time_integrate  ...
    CubQMCMLCont (StoppingCriterion Object)
        rmse_tol        0.019
        n_init          2^(8)
        n_max           10000000000
        replications    2^(5)
        levels_min      2^(1)
        levels_max      10
        n_tols          10
        tol_mult        1.668
        theta_init      2^(-1)
        theta           2^(-3)
    MLCallOptions (AbstractIntegrand Object)
        option          european
        sigma           0.200
        k               100
        r               0.050
        t               1
        b               85
        level           0
    Gaussian (AbstractTrueMeasure Object)
        mean            0
        covariance      1
        decomp_type     PCA
    Lattice (AbstractDiscreteDistribution Object)
        d               1
        dvec            0
        randomize       SHIFT
        order           NATURAL
        gen_vec         1
        entropy         7
        spawn_key       ()

    References:
        
        [1] https://github.com/PieterjanRobbe/MultilevelEstimators.jl

    """

    def __init__(self, integrand, abs_tol=.05, alpha=.01, rmse_tol=None, n_init=256., n_max=1e10, 
        replications=32., levels_min=2, levels_max=10, n_tols=10, tol_mult=100**(1/9), theta_init=0.5):
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
            replications (int): number of replications on each level
            levels_min (int): minimum level of refinement >= 2
            levels_max (int): maximum level of refinement >= Lmin
            n_tols (int): number of coarser tolerances to run
            tol_mult (float): coarser tolerance multiplication factor
            theta_init (float) : initial error splitting constant

        """
        self.parameters = ['rmse_tol','n_init','n_max','replications','levels_min',
            'levels_max','n_tols','tol_mult','theta_init','theta']
        # initialization
        if rmse_tol:
            self.target_tol = float(rmse_tol)
        else: # use absolute tolerance
            self.target_tol =  float(abs_tol) / norm.ppf(1-alpha/2)
        self.n_init = float(n_init)
        self.n_max = float(n_max)
        self.replications = float(replications)
        self.levels_min = levels_min
        self.levels_max = levels_max
        self.theta_init = theta_init
        self.theta = theta_init
        self.n_tols = n_tols
        self.tol_mult = tol_mult
        # QMCPy Objs
        self.integrand = integrand
        self.true_measure = self.integrand.true_measure
        self.discrete_distrib = self.integrand.discrete_distrib
        # Verify Compliant Construction
        allowed_levels = ['adaptive-multi']
        allowed_distribs = [LD]
        allow_vectorized_integrals = False
        super(CubQMCMLCont,self).__init__(allowed_levels, allowed_distribs, allow_vectorized_integrals)

    def integrate(self):
        # Construct AccumulateData Object to House Integration Data
        self.data = MLQMCData(self, self.integrand, self.true_measure, self.discrete_distrib,
            self.levels_min, self.levels_max, self.n_init, self.replications)
        # Loop over coarser tolerances
        for t in range(self.n_tols):
            self.rmse_tol = self.tol_mult**(self.n_tols-t-1)*self.target_tol # Set new target tolerance
            self._integrate()
        return self.data.solution,self.data

    def _integrate(self):
        """ See abstract method. """
        t_start = time()
        #self.theta = self.theta_init
        self.data.levels = int(self.levels_min+1)

        converged = False
        while not converged:
            # Ensure that we have samples on the finest level
            self.data.update_data()
            self._update_theta()

            while self._varest() > (1-self.theta)*self.rmse_tol**2:
                efficient_level = np.argmax(self.data.var_cost_ratio_level[:self.data.levels])
                self.data.eval_level[efficient_level] = True

                # Check if over sample budget
                total_next_samples = (self.data.replications*self.data.eval_level*self.data.n_level*2).sum()
                if (self.data.n_total + total_next_samples) > self.n_max:
                    warning_s = """
                    Already generated %d samples.
                    Trying to generate %d new samples, which would exceed n_max = %d.
                    Stopping integration process.
                    Note that error tolerances may no longer be satisfied""" \
                    % (int(self.data.n_total), int(total_next_samples), int(self.n_max))
                    warnings.warn(warning_s, MaxSamplesWarning)
                    self.data.time_integrate += time() - t_start
                    return

                self.data.update_data()
                self._update_theta()

            # Check for convergence
            converged = self._rmse() < self.rmse_tol
            if not converged:
                if self.data.levels == self.levels_max:
                    warnings.warn(
                        'Failed to achieve weak convergence. levels == levels_max.',
                        MaxLevelsWarning)
                    converged = True
                else:
                    self.data._add_level()

        self.data.time_integrate = time() - t_start
    
    def set_tolerance(self, abs_tol=None, alpha=.01, rmse_tol=None):
        """
        See abstract method. 
        
        Args:
            integrand (AbstractIntegrand): integrand with multi-level g method
            abs_tol (float): absolute tolerance. Reset if supplied, ignored if not. 
            alpha (float): uncertainty level.
                If rmse_tol not supplied, then rmse_tol = abs_tol/norm.ppf(1-alpha/2)
            rel_tol (float): relative tolerance. Reset if supplied, ignored if not.
                Takes priority over absolute tolerance and alpha if supplied.
        """
        if rmse_tol != None:
            self.rmse_tol = float(rmse_tol)
        elif abs_tol != None:
            self.rmse_tol = (float(abs_tol) / norm.ppf(1-alpha/2.))

    def _update_theta(self):
        """Update error splitting parameter"""
        max_levels = len(self.data.n_level)
        A = np.ones((2,2))
        A[:,0] = range(max_levels-2, max_levels)
        y = np.ones(2)
        y[0] = np.log2(abs(self.data.mean_level_reps[max_levels-2].mean()))
        y[1] = np.log2(abs(self.data.mean_level_reps[max_levels-1].mean()))
        x = np.linalg.lstsq(A, y, rcond=None)[0]
        alpha = max(.5,-x[0])
        real_bias = 2**(x[1]+max_levels*x[0]) / (2**alpha - 1)
        self.theta = max(0.01, min(0.125, (real_bias/self.rmse_tol)**2))

    def _rmse(self):
        """Returns an estimate for the root mean square error"""
        return np.sqrt(self._mse())

    def _mse(self):
        """Returns an estimate for the mean square error"""
        return (1-self.theta)*self._varest() + self.theta*self.data.bias_estimate**2

    def _varest(self):
        """Returns the variance of the estimator"""
        return self.data.var_level[:self.data.levels].sum()
