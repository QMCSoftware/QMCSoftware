from ._stopping_criterion import StoppingCriterion
from ..accumulate_data import MLQMCData
from ..discrete_distribution import Lattice
from ..true_measure import Gaussian
from ..integrand import MLCallOptions
from ..util import MaxSamplesWarning, ParameterError, MaxLevelsWarning
from numpy import *
from scipy.stats import norm
from time import time
import warnings


class CubQMCML(StoppingCriterion):
    """
    Stopping criterion based on multi-level quasi-Monte Carlo.

    >>> mlco = MLCallOptions(Lattice(seed=7))
    >>> sc = CubQMCML(mlco,abs_tol=.05)
    >>> solution,data = sc.integrate()
    >>> solution
    10.433...
    >>> data
    Solution: 10.4332        
    MLCallOptions (Integrand Object)
        option          european
        sigma           0.200
        k               100
        r               0.050
        t               1
        b               85
    Lattice (DiscreteDistribution Object)
        d               2^(5)
        randomize       1
        order           natural
        seed            87657
        mimics          StdUniform
    Gaussian (TrueMeasure Object)
        mean            0
        covariance      1
        decomp_type     pca
    CubQMCML (StoppingCriterion Object)
        rmse_tol        0.019
        n_init          2^(8)
        n_max           10000000000
        replications    2^(5)
    MLQMCData (AccumulateData Object)
        levels          6
        dimensions      [ 1.  2.  4.  8. 16. 32.]
        n_level         [4096.  256.  256.  256.  256.  256.]
        mean_level      [10.053  0.182  0.102  0.054  0.028  0.014]
        var_level       [6.399e-05 6.275e-05 2.998e-05 1.024e-05 3.161e-06 1.411e-06]
        bias_estimate   0.008
        n_total         172032
        time_integrate  ...
    
    References:
        
        [1] M.B. Giles and B.J. Waterhouse. 'Multilevel quasi-Monte Carlo path simulation'.
        pp.165-181 in Advanced Financial Modelling, in Radon Series on Computational and Applied Mathematics,
        de Gruyter, 2009. http://people.maths.ox.ac.uk/~gilesm/files/radon.pdf
    """

    def __init__(self, integrand, abs_tol=.05, alpha=.01, rmse_tol=None, n_init=256., n_max=1e10, 
        replications=32., levels_min=2, levels_max=10):
        """
        Args:
            integrand (Integrand): integrand with multi-level g method
            abs_tol (float): absolute tolerance
            alpha (float): uncertaintly level.
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
        self.n_init = float(n_init)
        self.n_max = float(n_max)
        self.replications = float(replications)
        self.levels_min = levels_min
        self.levels_max = levels_max
        # QMCPy Objs
        self.integrand = integrand
        self.true_measure = self.integrand.true_measure
        self.discrete_distrib = self.integrand.discrete_distrib
        # Verify Compliant Construction
        allowed_levels = ['adaptive-multi']
        allowed_distribs = ["Lattice", "Sobol","Halton"]
        super(CubQMCML,self).__init__(allowed_levels, allowed_distribs)

    def integrate(self):
        """ See abstract method. """
        # Construct AccumulateData Object to House Integration Data
        self.data = MLQMCData(self, self.integrand, self.true_measure, self.discrete_distrib,
            self.levels_min, self.n_init, self.replications)
        t_start = time()
        while True:
            self.data.update_data()
            if self.data.var_level.sum() > (self.rmse_tol**2/2.):
                # double N_l on level with largest V_l/(2^l*N_l)
                efficient_level = argmax(self.data.var_cost_ratio_level)
                self.data.eval_level[efficient_level] = True
            elif self.data.bias_estimate > (self.rmse_tol/sqrt(2.)):
                if self.data.levels == self.levels_max + 1:
                        warnings.warn("Failed to achieve weak convergence. levels == levels_max.", MaxLevelsWarning)
                # add another level
                self.data._add_level()
            else:
                # both conditions met
                break
            total_next_samples = (self.data.replications*self.data.eval_level*self.data.n_level*2).sum()
            if (self.data.n_total + total_next_samples) > self.n_max:
                warning_s = """
                Alread generated %d samples.
                Trying to generate %d new samples, which would exceed n_max = %d.
                Stopping integration process.
                Note that error tolerances may no longer be satisfied""" \
                % (int(self.data.n_total), int(total_next_samples), int(self.n_max))
                warnings.warn(warning_s, MaxSamplesWarning)
                break
        self.data.time_integrate = time() - t_start
        return self.data.solution,self.data
    
    def set_tolerance(self, abs_tol=None, alpha=.01, rmse_tol=None):
        """
        See abstract method. 
        
        Args:
            integrand (Integrand): integrand with multi-level g method
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
