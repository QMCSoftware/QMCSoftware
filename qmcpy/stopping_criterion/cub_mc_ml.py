from ._stopping_criterion import StoppingCriterion
from ..accumulate_data import MLMCData
from ..discrete_distribution import IIDStdGaussian
from ..true_measure import Gaussian
from ..integrand import MLCallOptions
from ..util import MaxSamplesWarning, ParameterError, MaxLevelsWarning, ParameterWarning
from numpy import *
from scipy.stats import norm
from time import time
import warnings


class CubMCML(StoppingCriterion):
    """
    Stopping criterion based on multi-level monte carlo.
    
    >>> mlco = MLCallOptions(Gaussian(IIDStdGaussian(seed=7)))
    >>> sc = CubMCML(mlco,abs_tol=.05)
    >>> solution,data = sc.integrate()
    >>> solution
    10.445441712933263
    >>> data
    Solution: 10.4454        
    MLCallOptions (Integrand Object)
        option          european
        sigma           0.200
        k               100
        r               0.050
        t               1
        b               85
    IIDStdGaussian (DiscreteDistribution Object)
        dimension       2^(6)
        seed            7
        mimics          StdGaussian
    Gaussian (TrueMeasure Object)
        mean            0
        covariance      1
        decomp_type     pca
    CubMCML (StoppingCriterion Object)
        rmse_tol        0.019
        n_init          2^(8)
        levels_min      2^(1)
        levels_max      10
        theta           2^(-2)
    MLMCData (AccumulateData Object)
        levels          7
        dimensions      [ 1.  2.  4.  8. 16. 32. 64.]
        n_level         [7.804e+05 1.533e+04 6.633e+03 2.077e+03 7.560e+02 2.730e+02 9.600e+01]
        mean_level      [1.006e+01 1.848e-01 1.014e-01 5.138e-02 2.472e-02 1.452e-02 7.657e-03]
        var_level       [1.963e+02 1.515e-01 4.124e-02 1.109e-02 2.901e-03 6.799e-04 1.848e-04]
        cost_per_sample [ 1.  2.  4.  8. 16. 32. 64.]
        n_total         805984
        alpha           0.927
        beta            1.946
        gamma           1.000
        time_integrate  ...

    Original Implementation:

        http://people.maths.ox.ac.uk/~gilesm/mlmc/#MATLAB

    References:
        
        [1] M.B. Giles. 'Multi-level Monte Carlo path simulation'. 
        Operations Research, 56(3):607-617, 2008.
        http://people.maths.ox.ac.uk/~gilesm/files/OPRE_2008.pdf.
    """

    parameters = ['rmse_tol','n_init','levels_min','levels_max','theta']

    def __init__(self, integrand, abs_tol=.05, alpha=.01, rmse_tol=None, n_init=256., n_max=1e10, 
        levels_min=2., levels_max=10., alpha0=-1., beta0=-1., gamma0=-1.):
        """
        Args:
            integrand (Integrand): integrand with multi-level g method
            abs_tol (float): absolute tolerance. Reset if supplied, ignored if not. 
            alpha (float): uncertaintly level.
                If rmse_tol not supplied, then rmse_tol = abs_tol/norm.ppf(1-alpha/2)
            rel_tol (float): relative tolerance. Reset if supplied, ignored if not.
                Takes priority over aboluste tolerance and alpha if supplied. 
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
        self.n_init = float(n_init)
        self.n_max = float(n_max)
        self.levels_min = float(levels_min)
        self.levels_max = float(levels_max)
        self.theta = 0.25
        self.integrand = integrand
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.gamma0 = gamma0
        # Verify Compliant Construction
        distribution = integrand.measure.distribution
        allowed_levels = ['adaptive-multi']
        allowed_distribs = ["IIDStdUniform", "IIDStdGaussian", "CustomIIDDistribution"]
        super(CubMCML,self).__init__(distribution, integrand, allowed_levels, allowed_distribs)
    
    def integrate(self):
        """ See abstract method. """
         # Construct AccumulateData Object to House Integration Data
        self.data = MLMCData(self, self.integrand, self.levels_min, self.n_init, self.alpha0, self.beta0, self.gamma0)
        t_start = time()
        while self.data.diff_n_level.sum() > 0:
            self.data.update_data()
            self.data.n_total += self.data.diff_n_level.sum()
            # set optimal number of additional samples
            n_samples = self._get_next_samples()
            self.data.diff_n_level = maximum(0, n_samples-self.data.n_level)
            # check if over sample budget
            if (self.data.n_total + self.data.diff_n_level.sum()) > self.n_max:
                warning_s = """
                Alread generated %d samples.
                Trying to generate %d new samples, which would exceed n_max = %d.
                Stopping integration process.
                Note that error tolerances may no longer be satisfied""" \
                % (int(self.data.n_total), int(self.data.diff_n_level.sum()), int(self.n_max))
                warnings.warn(warning_s, MaxSamplesWarning)
                break
            # if (almost) converged, estimate remaining error and decide 
            # whether a new level is required
            if (self.data.diff_n_level > 0.01*self.data.n_level).sum() == 0:
                range_ = arange(minimum(2.,self.data.levels-1)+1)
                idx = (self.data.levels-range_).astype(int)
                rem = (self.data.mean_level[idx] / 
                        2.**(range_*self.data.alpha)).max() / (2.**self.data.alpha - 1)
                # rem = ml(l+1) / (2^alpha - 1)
                if rem > sqrt(self.theta)*self.rmse_tol:
                    if self.data.levels == self.levels_max:
                        warnings.warn(
                            'Failed to achieve weak convergence. levels == levels_max.',
                            MaxLevelsWarning)
                    else:
                        self.data.levels += 1
                        self.data.dimensions = hstack((self.data.dimensions,0))
                        self.data.var_level = hstack((self.data.var_level,
                            self.data.var_level[-1] / 2**self.data.beta))
                        self.data.cost_per_sample = hstack((self.data.cost_per_sample, 
                            self.data.cost_per_sample[-1] * 2**self.data.gamma))
                        self.data.n_level = hstack((self.data.n_level, 0.))
                        self.data.sum_level = hstack((self.data.sum_level,
                            zeros((2,1))))
                        self.data.cost_level = hstack((self.data.cost_level, 0.))
                        n_samples = self._get_next_samples()
                        self.data.diff_n_level = maximum(0., n_samples-self.data.n_level)
        # finally, evaluate multilevel estimator
        self.data.solution = (self.data.sum_level[0,:]/self.data.n_level).sum()
        self.data.levels += 1
        self.data.time_integrate = time() - t_start
        return self.data.solution,self.data
    
    def _get_next_samples(self):
        ns = ceil( sqrt(self.data.var_level/self.data.cost_per_sample) * 
                sqrt(self.data.var_level*self.data.cost_per_sample).sum() / 
                ((1-self.theta)*self.rmse_tol**2) )
        return ns
    
    def set_tolerance(self, abs_tol=None, alpha=.01, rmse_tol=None):
        """
        See abstract method. 
        
        Args:
            abs_tol (float): absolute tolerance. Reset if supplied, ignored if not. 
            alpha (float): uncertaintly level.
                If rmse_tol not supplied, then rmse_tol = abs_tol/norm.ppf(1-alpha/2)
            rel_tol (float): relative tolerance. Reset if supplied, ignored if not.
                Takes priority over aboluste tolerance and alpha if supplied. 
        """
        if rmse_tol != None:
            self.rmse_tol = float(rmse_tol)
        elif abs_tol != None:
            self.rmse_tol = (float(abs_tol) / norm.ppf(1-alpha/2.))