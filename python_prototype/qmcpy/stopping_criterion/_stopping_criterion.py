""" Definition for abstract class StoppingCriterion """

from abc import ABC, abstractmethod
from numpy import floor

from .._util import DistributionCompatibilityError, univ_repr, ParameterError


class StoppingCriterion(ABC):
    """ Decide when to stopping_criterion """

    def __init__(self, distribution, allowed_distribs):
        """
        Args:
            distribution (DiscreteDistribution): an instance of DiscreteDistribution
            allowed_distribs: distribution's compatible with the StoppingCriterion
        """
        super().__init__()
        if type(distribution).__name__ not in allowed_distribs:
            error_message = type(self).__name__  \
                + " only accepts distributions:" \
                + str(allowed_distribs)
            raise DistributionCompatibilityError(error_message)
        string_prefix = 'A concrete implementation of Stopping Criterion must have '
        if not hasattr(self, 'abs_tol'):
            raise ParameterError(string_prefix+'self.abs_tol (absolute tolerance)')
        if not hasattr(self, 'rel_tol'):
            raise ParameterError(string_prefix+'self.rel_tol (relative tolerance)')
        if not hasattr(self, 'n_init'):
            raise ParameterError(string_prefix+'self.n_init (initial sample size)')
        if not hasattr(self, 'n_max'):
            raise ParameterError(string_prefix+'self.n_max (maximum total samples)')
        if not hasattr(self, 'alpha'):
            raise ParameterError(string_prefix+'self.alpha (uncertainty level)')
        if not hasattr(self, 'inflate'):
            raise ParameterError(string_prefix+'self.inflate (inflation factor)')
        if not hasattr(self, 'stage'):
            raise ParameterError(string_prefix+'self.stage (stage of the computation)')

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
