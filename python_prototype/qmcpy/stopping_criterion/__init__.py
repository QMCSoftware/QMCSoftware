""" Definition for abstract class StoppingCriterion """

from abc import ABC, abstractmethod
import warnings
from numpy import floor

from qmcpy._util import MaxSamplesWarning
from .._util import DistributionCompatibilityError, univ_repr


class StoppingCriterion(ABC):
    """ Decide when to stopping_criterion """

    def __init__(self, distribution, allowed_distribs, abs_tol, rel_tol, n_init, n_max):
        """
        Args:
            distribution (DiscreteDistribution): an instance of DiscreteDistribution
            allowed_distribs: distribution's compatible with the StoppingCriterion
            abs_tol (float): absolute error tolerance
            rel_tol (float): relative error tolerance
            n_init (int): initial number of samples
            n_max (int): maximum number of samples
        """
        if type(distribution).__name__ not in allowed_distribs:
            error_message = type(self).__name__  \
                + " only accepts distributions:" \
                + str(allowed_distribs)
            raise DistributionCompatibilityError(error_message)
        super().__init__()
        self.abs_tol = abs_tol if abs_tol else 1e-2
        self.rel_tol = rel_tol if rel_tol else 0
        self.n_init = n_init if n_init else 1024
        self.n_max = n_max if n_max else 1e8
        self.alpha = None
        self.inflate = None

    @abstractmethod
    def stop_yet(self):
        """ Determine when to stopping_criterion """

    def check_n(self,n_next):
        if self.data.n_total + n_next.sum() <= self.n_max: 
            # new samples will not go over the max
            return 0,n_next
        # cannot generate this many new samples
        warning_s = """
        Alread generated %d samples.
        Trying to generate %s new samples, which exceeds n_max = %d.
        The number of new samples will be decrease proportionally for each integrand.
        Note that error tolerances may no longer be satisfied""" \
        %(int(self.data.n_total),str(n_next),int(self.n_max))
        warnings.warn(warning_s,MaxSamplesWarning)
        # decrease n proportionally for each integrand
        n_decease = self.data.n_total + n_next.sum() - self.n_max
        dec_prop = n_decease/n_next.sum()
        return 1,floor(n_next-n_next*dec_prop)

    def summarize(self):
        """ Print important attribute values """
        header_fmt = "%s (%s)\n"
        item_i = "%25s: %d\n"
        item_f = "%25s: %-15.4f\n"
        obj_name = "StoppingCriterion Object"
        attrs_vals_str = header_fmt % (type(self).__name__, obj_name)
        attrs_vals_str += item_f % ("abs_tol", self.abs_tol)
        attrs_vals_str += item_f % ("rel_tol", self.rel_tol)
        attrs_vals_str += item_i % ("n_max", self.n_max)
        attrs_vals_str += item_f % ("inflate", self.inflate)
        attrs_vals_str += item_f % ("alpha", self.alpha)
        print(attrs_vals_str[:-2])

    def __repr__(self):
        return univ_repr(self)


# API
from .clt_rep import CLTRep
from .clt import CLT
from .mean_mc_g import MeanMC_g
