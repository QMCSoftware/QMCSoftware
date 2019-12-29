""" Definition for CLTRep, a concrete implementation of StoppingCriterion """

from ._stopping_criterion import StoppingCriterion
from .._util import MaxSamplesWarning, NotYetImplemented, ParameterWarning
from ..accum_data import MeanVarDataRep

import warnings
from numpy import array, tile, sqrt, log
from scipy.stats import norm


class CLTRep(StoppingCriterion):
    """ Stopping criterion based on var(stream_1_estimate, ..., stream_16_estimate) < errorTol """

    def __init__(self, discrete_distrib, true_measure,
                 replications=16, inflate=1.2, alpha=0.01,
                 abs_tol=1e-2, rel_tol=0,
                 n_init=32, n_max=2**30):
        """
        Args:
            discrete_distrib
            true_measure (DiscreteDistribution): an instance of DiscreteDistribution
            replications (int): number of random nxm matrices to generate
            inflate (float): inflation factor when estimating variance
            alpha (float): significance level for confidence interval
            abs_tol (float): absolute error tolerance
            rel_tol (float): relative error tolerance
            n_max (int): maximum number of samples
        """
        # Input Checks
        levels = len(true_measure)
        if levels != 1:
            raise NotYetImplemented('''
                CLTRep not implemented for multi-level problems.
                Use CLT stopping criterion with an iid distribution for multi-level problems ''')
        if (log(n_init) / log(2)) % 1 != 0:
            warning_s = ' n_init must be a power of 2. Using n_init = 32'
            warnings.warn(warning_s, ParameterWarning)
            n_init = 32
        # Set Attributes
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.n_max = n_max
        self.alpha = alpha
        self.inflate = inflate
        self.stage = "begin"
        # Construct Data Object to House Integration data
        self.data = MeanVarDataRep(levels, n_init, replications)
        # Verify Compliant Construction
        allowed_distribs = ["Lattice", "Sobol"]
        super().__init__(discrete_distrib, allowed_distribs)

    def stop_yet(self):
        """ Determine when to stop """
        sighat_up = sqrt((self.data.sighat**2).sum() /
                         len(self.data.sighat)) * self.inflate
        if sighat_up > max(self.abs_tol, abs(self.data.solution) * self.rel_tol):
            # Not sufficiently estimated
            if 2 * self.data.n.sum() > self.n_max:
                # doubling samples would go over n_max
                warning_s = """
                Alread generated %d samples.
                Trying to generate %s new samples, which exceeds n_max = %d.
                No more samples will be generated.
                Note that error tolerances may no longer be satisfied""" \
                % (int(self.data.n_total.sum()), str(self.data.n), int(self.n_max))
                warnings.warn(warning_s, MaxSamplesWarning)
                self.stage = 'done'
            else:
                self.data.n *= 2  # double n for next sample
        else:
            self.stage = 'done'  # sufficiently estimated
        if self.stage == 'done':
            self.data.n_total = self.data.n_total.sum()
            err_bar = -norm.ppf(self.alpha / 2) * self.inflate \
                * (self.data.sighat ** 2 / self.data.n).sum(0) ** 0.5
            self.data.confid_int = self.data.solution + \
                err_bar * array([-1, 1])  # CLT confidence interval
            self.stage = "done"
