""" Definition for abstract class StoppingCriterion """

from abc import ABC, abstractmethod

from .._util import DistributionCompatibilityError, univ_repr


class StoppingCriterion(ABC):
    """ Decide when to stop """

    def __init__(self, distribution, allowed_distribs, abs_tol, rel_tol,
                 n_init, n_max):
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
    def stop_yet(self):  # accum_data = summary of data computed already
        """
        Determine when to stop
        """

    def __repr__(self):
        return univ_repr(self)

    def summarize(self):
        """Print important attribute values.
        """
        header_fmt = "%s (%s)\n"
        item_i = "%25s: %d\n"
        item_f = "%25s: %-15.4f\n"

        attrs_vals_str = ""
        attrs_vals_str += header_fmt % (type(self).__name__,
                                        "StoppingCriterion Object")
        attrs_vals_str += item_f % ("abs_tol", self.abs_tol)
        attrs_vals_str += item_f % ("rel_tol", self.rel_tol)
        attrs_vals_str += item_i % ("n_max", self.n_max)
        attrs_vals_str += item_f % ("inflate", self.inflate)
        attrs_vals_str += item_f % ("alpha", self.alpha)

        print(attrs_vals_str[:-2])


# API
from .clt_rep import CLTRep
from .clt import CLT