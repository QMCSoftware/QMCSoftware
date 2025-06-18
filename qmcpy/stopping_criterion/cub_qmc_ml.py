from .abstract_stopping_criterion import AbstractStoppingCriterion
from ..accumulate_data import AccumulateData
from ..discrete_distribution import DigitalNetB2,Lattice,Halton
from ..discrete_distribution.abstract_discrete_distribution import AbstractLDDiscreteDistribution
from ..true_measure import Gaussian
from ..integrand import MLCallOptions
from ..util import MaxSamplesWarning, ParameterError, MaxLevelsWarning
import numpy as np
from scipy.stats import norm
from time import time
import warnings


class CubQMCML(AbstractStoppingCriterion):
    """
    Stopping criterion based on multi-level quasi-Monte Carlo.

    >>> mlco = MLCallOptions(Lattice(seed=7,replications=32))
    >>> sc = CubQMCML(mlco,abs_tol=.075)
    >>> solution,data = sc.integrate()
    >>> data
    AccumulateData (AccumulateData)
        solution        10.418
        n_total         98304
        levels          5
        n_level         [2048  256  256  256  256]
        mean_level      [10.053  0.183  0.102  0.053  0.028]
        var_level       [2.454e-04 5.669e-05 2.597e-05 1.360e-05 7.235e-06]
        bias_estimate   0.016
        time_integrate  0.050
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
            replications (int): number of replications on each level
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
        data = AccumulateData(parameters=[
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
    
    def update_data(self, data):
        # update sample sums
        for l in range(data.levels):
            if not data.eval_level[l]:
                # nothing to do on this level
                continue
            if l==len(data.level_integrands):
                # haven't spawned this level's integrand yet
                data.level_integrands += self.integrand.spawn(levels=l)
            # reset dimension
            n_max = self.n_init if data.n_level[l]==0 else 2*data.n_level[l]
            integrand_l = data.level_integrands[l]
            samples = integrand_l.discrete_distrib.gen_samples(n_min=data.n_level[l],n_max=n_max)
            integrand_l.f(samples).squeeze()
            prev_sum = data.mean_level_reps[l]*data.n_level[l]
            data.mean_level_reps[l] = (integrand_l.sums[...,0]+prev_sum)/float(n_max)
            data.cost_level[l] = data.cost_level[l] + integrand_l.cost
            data.n_level[l] = n_max
            data.mean_level[l] = data.mean_level_reps[l].mean()
            data.var_level[l] = data.mean_level_reps[l].var()
            cost_per_sample = data.cost_level[l]/data.n_level[l]/self.replications
            data.var_cost_ratio_level[l] = data.var_level[l]/cost_per_sample
        self._update_bias_estimate(data)
        data.n_total = self.replications*data.n_level.sum()
        data.solution = data.mean_level.sum()
        data.eval_level[:] = False # Reset active levels

    def _update_bias_estimate(self, data):
        A = np.ones((2,2))
        A[:,0] = range(data.levels-2, data.levels)
        y = np.ones(2)
        y[0] = np.log2(abs(data.mean_level_reps[data.levels-2].mean()))
        y[1] = np.log2(abs(data.mean_level_reps[data.levels-1].mean()))
        x = np.linalg.lstsq(A,y,rcond=None)[0]
        alpha = max(.5,-x[0])
        data.bias_estimate = 2**(x[1]+data.levels*x[0]) / (2**alpha - 1)
    
    def _add_level(self, data):
        # Add another level to relevant attributes.
        data.levels += 1
        if data.levels > len(data.n_level):
            data.n_level = np.hstack((data.n_level,0))
            data.eval_level = np.hstack((data.eval_level,True))
            data.mean_level_reps = np.vstack((data.mean_level_reps,np.zeros(int(self.replications))))
            data.mean_level = np.hstack((data.mean_level,0))
            data.var_level = np.hstack((data.var_level,np.inf))
            data.cost_level = np.hstack((data.cost_level,0))
            data.var_cost_ratio_level = np.hstack((data.var_cost_ratio_level,np.inf))

    def _add_level_MLMC(self, data):
        # Add another level to relevant attributes.
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
