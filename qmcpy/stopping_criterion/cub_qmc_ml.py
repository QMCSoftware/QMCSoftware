from ._stopping_criterion import StoppingCriterion
from ..accumulate_data import MLQMCData
from ..util import MaxSamplesWarning, ParameterError
from numpy import argmax, sqrt
from scipy.stats import norm
from time import perf_counter
import warnings


class CubQmcMl(StoppingCriterion):
    """
    Stopping criterion based on multi-level quasi-monte carlo
    
    Reference:
        M.B. Giles and B.J. Waterhouse. 'Multilevel quasi-Monte Carlo path simulation'.
        pp.165-181 in Advanced Financial Modelling, in Radon Series on Computational and Applied Mathematics,
        de Gruyter, 2009. http://people.maths.ox.ac.uk/~gilesm/files/radon.pdf
    """

    parameters = ['rmse_tol','n_init','n_max','replications']

    def __init__(self, integrand, abs_tol=.05, alpha=.01, rmse_tol=None, n_init=256, n_max=1e10, replications=32):
        """
        Args:
            integrand (Integrand): integrand with multi-level g method
            abs_tol (float): absolute tolerence
            alpha (float): uncertainty level 
            rmse_tol (float): root mean squared error
                If supplied (not None), then absolute tolerence and alpha are ignored
                in favor of the rmse tolerence
            n_max (int): maximum number of samples
            replications (int): number of replications on each level
        """
        # initialization
        self.rmse_tol = rmse_tol if rmse_tol else (abs_tol / norm.ppf(1-alpha/2))
        self.n_init = n_init
        self.n_max = n_max
        self.replications = replications
        # Verify Compliant Construction
        distribution = integrand.measure.distribution
        allowed_levels = 'multi'
        allowed_distribs = ["Lattice", "Sobol"]
        super().__init__(distribution, allowed_levels, allowed_distribs)
        # Construct AccumulateData Object to House Integration Data
        self.data = MLQMCData(self, integrand, self.n_init, self.replications)
    
    def integrate(self):
        """ See abstract method. """
        t_start = perf_counter()
        while True:
            self.data.update_data()
            self.data.eval_level[:] = False
            if self.data.var_level.sum() > (self.rmse_tol**2/2):
                # double N_l on level with largest V_l/(2^l*N_l)
                efficient_level = argmax(self.data.cost_level)
                self.data.eval_level[efficient_level] = True
            elif self.data.bias_estimate > (self.rmse_tol/sqrt(2)):
                # add another level
                self.data.add_level()
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
        self.data.time_integrate = perf_counter() - t_start
        return self.data.solution,self.data
    