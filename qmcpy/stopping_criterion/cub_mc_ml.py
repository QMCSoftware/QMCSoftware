from .abstract_stopping_criterion import AbstractStoppingCriterion
from ..accumulate_data import AccumulateData
from ..discrete_distribution import IIDStdUniform
from ..discrete_distribution.abstract_discrete_distribution import AbstractIIDDiscreteDistribution
from ..true_measure import Gaussian
from ..integrand import MLCallOptions
from ..util import MaxSamplesWarning, ParameterError, MaxLevelsWarning, ParameterWarning
import numpy as np
from scipy.stats import norm
from time import time
import warnings


class CubMCML(AbstractStoppingCriterion):
    """
    Stopping criterion based on multi-level monte carlo.
    
    >>> mlco = MLCallOptions(IIDStdUniform(seed=7))
    >>> sc = CubMCML(mlco,abs_tol=.1)
    >>> solution,data = sc.integrate()
    >>> data
    AccumulateData (AccumulateData)
        solution        10.410
        n_total         298638
        levels          5
        n_level         [2.874e+05 6.434e+03 2.936e+03 7.060e+02 2.870e+02]
        mean_level      [10.044  0.193  0.103  0.046  0.025]
        var_level       [1.950e+02 1.954e-01 4.324e-02 9.368e-03 2.924e-03]
        cost_per_sample [ 1.  2.  4.  8. 16.]
        alpha           1.003
        beta            2.039
        gamma           1
        time_integrate  ...
    CubMCML (AbstractStoppingCriterion)
        rmse_tol        0.039
        n_init          2^(8)
        levels_min      2^(1)
        levels_max      10
        theta           2^(-1)
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

    def __init__(self, integrand, abs_tol=.05, alpha=.01, rmse_tol=None, n_init=256., n_max=1e10, 
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
        self.n_init = float(n_init)
        self.n_max = float(n_max)
        self.levels_min = float(levels_min)
        self.levels_max = float(levels_max)
        self.theta = 0.5
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.gamma0 = gamma0
        # QMCPy Objs
        self.integrand = integrand
        self.true_measure = self.integrand.true_measure
        self.discrete_distrib = self.integrand.discrete_distrib
        super(CubMCML,self).__init__(allowed_levels=['adaptive-multi'],allowed_distribs=[AbstractIIDDiscreteDistribution], allow_vectorized_integrals=False)

    def integrate(self):
        t_start = time()
        data = AccumulateData(parameters=[
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
        data.n_level = np.zeros(data.levels+1)
        data.sum_level = np.zeros((2,data.levels+1))
        data.cost_level = np.zeros(data.levels+1)
        data.diff_n_level = np.tile(self.n_init,data.levels+1)
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
                        _add_level(data)
                        n_samples = self._get_next_samples(data)
                        data.diff_n_level = np.maximum(0., n_samples-data.n_level)
        # finally, evaluate multilevel estimator
        data.solution = (data.sum_level[0,:]/data.n_level).sum()
        data.levels += 1
        data.stopping_crit = self
        data.integrand = self.integrand
        data.true_measure = self.integrand.true_measure
        data.discrete_distrib = self.true_measure.discrete_distrib
        data.time_integrate = time()-t_start
        return data.solution,data
            
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

    def _get_next_samples(self, data):
        ns = np.ceil( np.sqrt(data.var_level/data.cost_per_sample) * 
                np.sqrt(data.var_level*data.cost_per_sample).sum() / 
                ((1-self.theta)*self.rmse_tol**2) )
        return ns

    def _update_data(self, data):
        for l in range(data.levels+1):
            if l==len(data.level_integrands):
                # haven't spawned this level's integrand yet
                data.level_integrands += self.integrand.spawn(levels=int(l))
            integrand_l = data.level_integrands[l]
            if data.diff_n_level[l] > 0:
                # evaluate integral at sampling points samples
                samples = integrand_l.discrete_distrib.gen_samples(n=data.diff_n_level[l])
                integrand_l.f(samples).squeeze()
                data.n_level[l] = data.n_level[l] + data.diff_n_level[l]
                data.sum_level[0,l] = data.sum_level[0,l] + integrand_l.sums[0]
                data.sum_level[1,l] = data.sum_level[1,l] + integrand_l.sums[1]
                data.cost_level[l] = data.cost_level[l] + integrand_l.cost
        # compute absolute average, variance and cost
        data.mean_level = np.absolute(data.sum_level[0,:data.levels+1]/data.n_level[:data.levels+1])
        data.var_level = np.maximum(0,data.sum_level[1,:data.levels+1]/data.n_level[:data.levels+1] - data.mean_level**2)
        data.cost_per_sample = data.cost_level[:data.levels+1]/data.n_level[:data.levels+1]
        # fix to cope with possible zero values for data.mean_level and data.var_level
        # (can happen in some applications when there are few samples)
        for l in range(2,data.levels+1):
            data.mean_level[l] = np.maximum(data.mean_level[l], .5*data.mean_level[l-1]/2**data.alpha)
            data.var_level[l] = np.maximum(data.var_level[l], .5*data.var_level[l-1]/2**data.beta)
        # use linear regression to estimate alpha, beta, gamma if not given
        a = np.ones((data.levels,2))
        a[:,0] = np.arange(1,data.levels+1)
        if self.alpha0 <= 0:
            x = np.linalg.lstsq(a,np.log2(data.mean_level[1:]),rcond=None)[0]
            data.alpha = np.maximum(.5,-x[0])
        if self.beta0 <= 0:
            x = np.linalg.lstsq(a,np.log2(data.var_level[1:]),rcond=None)[0]
            data.beta = np.maximum(.5,-x[0])
        if self.gamma0 <= 0:
            x = np.linalg.lstsq(a,np.log2(data.cost_per_sample[1:]),rcond=None)[0]
            data.gamma = np.maximum(.5,x[0])
        data.n_total = data.n_level.sum()

def _add_level(data):
    data.levels += 1
    if not len(data.n_level) > data.levels:
        data.mean_level = np.hstack((data.mean_level, data.mean_level[-1] / 2**data.alpha))
        data.var_level = np.hstack((data.var_level, data.var_level[-1] / 2**data.beta))
        data.cost_per_sample = np.hstack((data.cost_per_sample, data.cost_per_sample[-1] * 2**data.gamma))
        data.n_level = np.hstack((data.n_level, 0.))
        data.sum_level = np.hstack((data.sum_level,np.zeros((2,1))))
        data.cost_level = np.hstack((data.cost_level, 0.))
    else:
        data.mean_level = np.absolute(data.sum_level[0,:data.levels+1]/data.n_level[:data.levels+1])
        data.var_level = np.maximum(0,data.sum_level[1,:data.levels+1]/data.n_level[:data.levels+1] - data.mean_level**2)
        data.cost_per_sample = data.cost_level[:data.levels+1]/data.n_level[:data.levels+1]
