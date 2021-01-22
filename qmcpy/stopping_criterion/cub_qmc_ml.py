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

    >>> mlco = MLCallOptions(Gaussian(Lattice(seed=7)))
    >>> sc = CubQMCML(mlco,abs_tol=.05)
    >>> solution,data = sc.integrate()
    >>> solution
    10.442361016810949
    >>> data
    Solution: 10.4424        
    MLCallOptions (Integrand Object)
        option          european
        sigma           0.200
        k               100
        r               0.050
        t               1
        b               85
    Lattice (DiscreteDistribution Object)
        dimension       2^(6)
        randomize       1
        order           natural
        seed            985802
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
        levels          7
        dimensions      [ 1.  2.  4.  8. 16. 32. 64.]
        n_level         [4096.  512.  256.  256.  256.  256.  256.]
        mean_level      [1.005e+01 1.807e-01 1.033e-01 5.482e-02 2.823e-02 1.397e-02 7.290e-03]
        var_level       [8.376e-05 2.660e-05 1.911e-05 1.594e-05 3.660e-06 1.478e-06 3.424e-07]
        bias_estimate   0.007
        n_total         188416
        time_integrate  ...
    
    
    References:
        
        [1] M.B. Giles and B.J. Waterhouse. 'Multilevel quasi-Monte Carlo path simulation'.
        pp.165-181 in Advanced Financial Modelling, in Radon Series on Computational and Applied Mathematics,
        de Gruyter, 2009. http://people.maths.ox.ac.uk/~gilesm/files/radon.pdf
    """

    parameters = ['rmse_tol','n_init','n_max','replications']

    def __init__(self, integrand, abs_tol=.05, alpha=.01, rmse_tol=None, n_init=256., n_max=1e10, replications=32., levels_max=10, bias_estimator='giles', cost_method='sde'):
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
            levels_max (int): maximum level of refinement >= Lmin
            bias_estimator (str): bias estimation method (can be 'giles' [default] or 'as_mlmc')
            cost_method (str): cost estimation method (can be 'sde' [default] or 'general')
        """
        # initialization
        if rmse_tol:
            self.rmse_tol = float(rmse_tol)
        else: # use absolute tolerance
            self.rmse_tol =  float(abs_tol) / norm.ppf(1-alpha/2)
        self.n_init = float(n_init)
        self.n_max = float(n_max)
        self.replications = float(replications)
        self.integrand = integrand
        self.bias_estimator = bias_estimator
        self.cost_method = cost_method
        # Verify Compliant Construction
        distribution = integrand.measure.distribution
        allowed_levels = ['adaptive-multi']
        allowed_distribs = ["Lattice", "Sobol","Halton"]
        super(CubQMCML,self).__init__(distribution, integrand, allowed_levels, allowed_distribs)

    def integrate(self):
        """ See abstract method. """
        # Construct AccumulateData Object to House Integration Data
        self.data = MLQMCData(self, self.integrand, self.n_init, self.replications, self.bias_estimator, self.cost_method)
        t_start = time()
        while True:
            self.data.update_data()
            self.data.eval_level[:] = False
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
