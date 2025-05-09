from .abstract_stopping_criterion import AbstractStoppingCriterion
from ..accumulate_data import MLMCData
from ..discrete_distribution import IIDStdUniform
from ..discrete_distribution.abstract_discrete_distribution import AbstractIIDDiscreteDistribution
from ..true_measure import Gaussian
from ..integrand import MLCallOptions
from ..util import MaxSamplesWarning, ParameterError, MaxLevelsWarning, ParameterWarning
import numpy as np
from scipy.stats import norm
from time import time
import warnings


class CubMCMLCont(AbstractStoppingCriterion):
    """
    Stopping criterion based on continuation multi-level monte carlo.
    
    >>> mlco = MLCallOptions(IIDStdUniform(seed=7))
    >>> sc = CubMCMLCont(mlco,abs_tol=.05)
    >>> solution,data = sc.integrate()
    >>> data
    MLMCData (AccumulateData Object)
        solution        10.400
        n_total         1193331
        levels          2^(2)
        n_level         [1133772.   22940.    8676.    2850.]
        mean_level      [10.059  0.186  0.105  0.05 ]
        var_level       [1.959e+02 1.603e-01 4.567e-02 1.013e-02]
        cost_per_sample [1. 2. 4. 8.]
        alpha           0.942
        beta            1.992
        gamma           1.000
        time_integrate  ...
    CubMCMLCont (AbstractStoppingCriterion Object)
        rmse_tol        0.019
        n_init          2^(8)
        levels_min      2^(1)
        levels_max      10
        n_tols          10
        tol_mult        1.668
        theta_init      2^(-1)
        theta           2^(-1)
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
    IIDStdUniform (AbstractDiscreteDistribution Object)
        d               1
        entropy         7
        spawn_key       ()

    References:

        [1] https://github.com/PieterjanRobbe/MultilevelEstimators.jl
        
    """

    def __init__(self, integrand, abs_tol=.05, alpha=.01, rmse_tol=None, n_init=256., n_max=1e10, 
        levels_min=2, levels_max=10, n_tols=10, tol_mult=100**(1/9), theta_init=0.5):
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
            n_tols (int): number of coarser tolerances to run
            tol_mult (float): coarser tolerance multiplication factor
            theta_init (float) : initial error splitting constant

        """
        self.parameters = ['rmse_tol','n_init','levels_min','levels_max','n_tols',
            'tol_mult','theta_init','theta']
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
        self.n_init = float(n_init)
        self.n_max = float(n_max)
        self.levels_min = float(levels_min)
        self.levels_max = float(levels_max)
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
        allowed_distribs = [IID]
        allow_vectorized_integrals = False
        super(CubMCMLCont,self).__init__(allowed_levels, allowed_distribs, allow_vectorized_integrals)

    def integrate(self):
        # Construct AccumulateData Object to House Integration Data
        self.data = MLMCData(self, self.integrand, self.true_measure, self.discrete_distrib,
            self.levels_min, self.n_init, -1., -1., -1.)
        # Loop over coarser tolerances
        for t in range(self.n_tols):
            self.rmse_tol = self.tol_mult**(self.n_tols-t-1)*self.target_tol # Set new target tolerance
            self._integrate()
        return self.data.solution,self.data
    
    def _integrate(self):
        """ See abstract method. """
        t_start = time()
        self.theta = self.theta_init
        self.data.levels = int(self.levels_min)
        self.data.update_data() # Take warm-up samples if none have been taken so far

        converged = False
        while not converged:

            # Check if we already have samples at the finest level
            if not self.data.n_level[self.data.levels] > 0:
                # This takes n_init warm-up samples at the finest level
                self.data.diff_n_level = np.hstack((self.data.diff_n_level,self.n_init))
                self.data.update_data()
                # Alternatively, evaluate optimal number of samples and take between 2 and n_init samples
                #self.data.diff_n_level = self._get_next_samples()
                #self.data.diff_n_level[:self.data.levels] = 0
                #self.data.diff_n_level[self.data.levels] = max(3, min(self.n_init, self.data.diff_n_level[self.data.levels]))

            # Update splitting parameter
            self._update_theta()

            # Set optimal number of additional samples
            n_samples = self._get_next_samples()
            self.data.diff_n_level = np.maximum(0, n_samples-self.data.n_level[:self.data.levels+1])
            
            # Check if over sample budget
            if (self.data.n_total + self.data.diff_n_level.sum()) > self.n_max:
                warning_s = """
                Already generated %d samples.
                Trying to generate %d new samples, which would exceed n_max = %d.
                Stopping integration process.
                Note that error tolerances may no longer be satisfied""" \
                % (int(self.data.n_total), int(self.data.diff_n_level.sum()), int(self.n_max))
                warnings.warn(warning_s, MaxSamplesWarning)
                break

            # Take additional samples
            self.data.update_data()
            self.data.n_total += self.data.diff_n_level.sum()

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

        self.data.diff_n_level.fill(0)
        self.data.solution = (self.data.sum_level[0,:self.data.levels+1]/self.data.n_level[:self.data.levels+1]).sum()
        self.data.levels += 1
        self.data.time_integrate += time() - t_start
    
    def _get_next_samples(self):
        ns = np.ceil( np.sqrt(self.data.var_level/self.data.cost_per_sample) * 
                np.sqrt(self.data.var_level*self.data.cost_per_sample).sum() / 
                ((1-self.theta)*self.rmse_tol**2) )
        return ns
    
    def set_tolerance(self, abs_tol=None, alpha=.01, rmse_tol=None):
        """
        See abstract method. 
        
        Args:
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
        self.theta = max(0.01, min(0.5, (self._bias(len(self.data.n_level)-1)/self.rmse_tol)**2))

    def _rmse(self):
        """Returns an estimate for the root mean square error"""
        return np.sqrt(self._mse())

    def _mse(self):
        """Returns an estimate for the mean square error"""
        return (1-self.theta)*self._varest() + self.theta*self._bias(self.data.levels)**2

    def _varest(self):
        """Returns the variance of the estimator"""
        return (self.data.var_level/self.data.n_level[:self.data.levels+1]).sum()

    def _bias(self, level):
        """Returns an estimate for the bias"""
        mean_level = self.data.sum_level[0, :]/self.data.n_level
        A = np.ones((2,2))
        A[:,0] = range(level-1, level+1)
        y = np.log2(abs(mean_level[level-1:level+1]))
        x = np.linalg.lstsq(A, y, rcond=None)[0]
        alpha = max(.5,-x[0])
        return 2**(x[1]+(level+1)*x[0]) / (2**alpha - 1)
