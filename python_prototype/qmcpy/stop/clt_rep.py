""" Definition for CLTRep, a concrete implementation of StoppingCriterion """

from numpy import array, full, zeros
from scipy.stats import norm

from . import StoppingCriterion
from ..accum_data import MeanVarDataRep
from .._util import MaxSamplesWarning


class CLTRep(StoppingCriterion):
    """ Stopping criterion based on var(stream_1_estimate, ..., stream_16_estimate) < errorTol """

    def __init__(
        self,
        discrete_distrib,
        true_measure,
        n_streams=16,
        inflate=1.2,
        alpha=0.01,
        abs_tol=1e-2,
        rel_tol=0,
        n_init=1024,
        n_max=1e8,
    ):
        """
        Args:
            discrete_distrib
            true_measure (DiscreteDistribution): an instance of DiscreteDistribution
            n_streams (int): number of random nxm matrices to generate
            inflate (float): inflation factor when estimating variance
            alpha (float): significance level for confidence interval
            abs_tol (float): absolute error tolerance
            rel_tol (float): relative error tolerance
            n_init (int): initial number of samples
            n_max (int): maximum number of samples
        """
        allowed_distribs = ["Lattice","Sobol"]  # supported distributions
        super().__init__(
            discrete_distrib, allowed_distribs, abs_tol, rel_tol, n_init, n_max
        )
        self.inflate = inflate  # inflation factor
        self.alpha = alpha  # uncertainty level
        self.stage = "begin"
        # Construct Data Object
        n_integrands = len(true_measure)
        self.data = MeanVarDataRep(
            n_integrands, n_streams
        )  # house integration data
        self.data.n_prev = zeros(n_integrands)  # previous n used for each f
        self.data.n_next = full(
            n_integrands, self.n_init
        )  # next n to be used for each f

    def stop_yet(self):
        """ Determine when to stop """
        for i in range(self.data.n_integrands):
            if (
                self.data.sig2hat[i] < self.abs_tol
            ):  # Sufficient estimate for mean of f[i]
                self.data.flag[i] = 0  # Stop estimation of i_th f
            else:  # Double n for next sample
                self.data.n_prev[i] = self.data.n_next[i]
                self.data.n_next[i] = self.data.n_prev[i] * 2
        if self.data.flag.sum() == 0 or self.data.n_next.max() > self.n_max:
            # Stopping Criterion Met
            if self.data.n_next.max() > self.n_max:
                raise MaxSamplesWarning(
                    "Max number of samples used. Tolerance may not be met"
                )
            self.data.n_samples_total = self.data.n_next
            err_bar = (
                -norm.ppf(self.alpha / 2)
                * self.inflate
                * (self.data.sig2hat ** 2 / self.data.n_next).sum(0) ** 0.5
            )
            self.data.confid_int = self.data.solution + err_bar * array(
                [-1, 1]
            )  # CLT confidence interval
            self.stage = "done"  # finished with computation
