""" Definition for CLTRep, a concrete implementation of StoppingCriterion """

from ._stopping_criterion import StoppingCriterion
from ..accum_data import MeanVarDataRep
from ..distribution import Distribution
from ..util import MaxSamplesWarning, NotYetImplemented, ParameterWarning
from numpy import array, log2, sqrt
from scipy.stats import norm
import warnings


class CLTRep(StoppingCriterion):
    """
    Stopping criterion based on
    var(stream_1_estimate, ..., stream_16_estimate) < errorTol
    """

    parameters = ['inflate','alpha','abs_tol','rel_tol','n_init','n_max']

    def __init__(self, distribution, inflate=1.2, alpha=0.01,
                 abs_tol=1e-2, rel_tol=0, n_init=32, n_max=2**30):
        """
        Args:
            distribution (Distribution): an instance of Distribution
            inflate (float): inflation factor when estimating variance
            alpha (float): significance level for confidence interval
            abs_tol (float): absolute error tolerance
            rel_tol (float): relative error tolerance
            n_max (int): maximum number of samples
        """
        # Input Checks
        if not isinstance(distribution,Distribution):
            # must be a list of Distributions objects -> multilevel problem
            raise NotYetImplemented('''
                CLTRep not implemented for multi-level problems.
                Use CLT stopping criterion with an iid distribution for multi-level problems.''')
        if log2(n_init) % 1 != 0:
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
        self.data = MeanVarDataRep(n_init, distribution.replications)
        # Verify Compliant Construction
        allowed_distribs = ["Lattice", "Sobol"]
        super().__init__(distribution, allowed_distribs)

    def stop_yet(self):
        """ Determine when to stop """
        sighat_up = self.data.sighat * self.inflate
        tol_up = max(self.abs_tol, abs(self.data.solution) * self.rel_tol)
        if sighat_up > tol_up:
            # Not sufficiently estimated
            if 2 * self.data.n > self.n_max:
                # doubling samples would go over n_max
                warning_s = """
                Alread generated %d samples.
                Trying to generate %d new samples would exceeds n_max = %d.
                No more samples will be generated.
                Note that error tolerances may not be satisfied""" \
                % (int(self.data.n_total), int(self.data.n), int(self.n_max))
                warnings.warn(warning_s, MaxSamplesWarning)
                self.stage = 'done'
            else:
                self.data.n *= 2  # double n for next sample
        else:
            self.stage = 'done' 
        if self.stage == 'done':
            # sufficiently estimated -> CLT confidence interval
            z_star = -norm.ppf(self.alpha / 2)
            err_bar = z_star * self.inflate * self.data.sighat / sqrt(self.data.n)
            self.data.confid_int = self.data.solution +  err_bar * array([-1, 1])
