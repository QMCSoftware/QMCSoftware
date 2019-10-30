""" Definition for CLT, a concrete implementation of StoppingCriterion """

import warnings

from numpy import array, ceil, floor, maximum, tile
from scipy.stats import norm

from . import StoppingCriterion
from .._util import MaxSamplesWarning
from ..accum_data import MeanVarData


class CLT(StoppingCriterion):
    """ Stopping criterion based on the Central Limit Theorem (CLT) """

    def __init__(self, discrete_distrib, true_measure,
                 inflate=1.2, alpha=0.01,
                 abs_tol=1e-2, rel_tol=0,
                 n_init=1024, n_max=1e8):
        """
        Args:
            discrete_distrib
            true_measure: an instance of DiscreteDistribution
            inflate: inflation factor when estimating variance
            alpha: significance level for confidence interval
            abs_tol: absolute error tolerance
            rel_tol: relative error tolerance
            n_init: initial number of samples
            n_max: maximum number of samples
        """
        allowed_distribs = ["IIDStdUniform",
                            "IIDStdGaussian"]  # supported distributions
        super().__init__(discrete_distrib, allowed_distribs, abs_tol, rel_tol,
                         n_init, n_max)
        self.inflate = inflate  # inflation factor
        self.alpha = alpha  # uncertainty level
        self.stage = "sigma"  # compute standard deviation next
        # Construct Data Object
        n_integrands = len(true_measure)
        self.data = MeanVarData(n_integrands)  # house integration data
        self.data.n = tile(self.n_init, n_integrands).astype(float)  # next n for each integrand

    def stop_yet(self):
        """ Determine when to stop """
        if self.stage == "sigma":
            temp_a = (self.data.t_eval) ** 0.5
            # use cost of function values to decide how to allocate
            temp_b = (temp_a * self.data.sighat).sum(0)
            # samples for computation of the mean
            # n_mu_temp := n such that confidence intervals width and conficence will be satisfied
            n_mu_temp = ceil(temp_b * (-norm.ppf(self.alpha / 2) * self.inflate /
                                       max(self.abs_tol, self.data.solution * self.rel_tol)) ** 2
                             * (self.data.sighat / self.data.t_eval ** .5))
            # n_mu := n_mu_temp adjusted for previous n
            self.data.n_mu = maximum(self.data.n, n_mu_temp)
            self.data.n += self.data.n_mu
            if self.data.n_total + self.data.n.sum() > self.n_max:
                # cannot generate this many new samples
                warning_s = """
                Alread generated %d samples.
                Trying to generate %s new samples, which exceeds n_max = %d.
                The number of new samples will be decrease proportionally for each integrand.
                Note that error tolerances may no longer be satisfied""" \
                % (int(self.data.n_total), str(self.data.n), int(self.n_max))
                warnings.warn(warning_s, MaxSamplesWarning)
                # decrease n proportionally for each integrand
                n_decease = self.data.n_total + self.data.n.sum() - self.n_max
                dec_prop = n_decease / self.data.n.sum()
                self.data.n = floor(self.data.n - self.data.n * dec_prop)
            self.stage = "mu"  # compute sample mean next
        elif self.stage == "mu":
            err_bar = -norm.ppf(self.alpha / 2) * self.inflate \
                * (self.data.sighat ** 2 / self.data.n_mu).sum(0) ** 0.5
            self.data.confid_int = self.data.solution + err_bar * array([-1, 1])
            # CLT confidence interval
            self.stage = "done"  # finished with computation
