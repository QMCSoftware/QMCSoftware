""" Definition for abstract class StoppingCriterion """

from abc import ABC, abstractmethod
from numpy import floor

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

    def __repr__(self, attributes=[]):
        """
        Print important attribute values

        Args: 
            attributes (list): list of attributes to print

        Returns:
            string of self info
        """
        attributes = set(attributes + ['abs_tol', 'rel_tol', 'n_max', 'inflate', 'alpha'])
        return univ_repr(self, "Stopping Criterion", attributes)
