""" Definition for abstract class StoppingCriterion """

from .._util import DistributionCompatibilityError, univ_repr, ParameterError

from abc import ABC, abstractmethod
from numpy import floor


class StoppingCriterion(ABC):
    """
    Decide when to stopping_criterion

    Attributes:
        abs_tol: absolute error tolerance
        rel_tol: relative error tolerance
        n_max: maximum number of samples
        alpha: significance level for confidence interval
        inflate: inflation factor when estimating variance
        stage: stage of the computation
    """

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
        prefix = 'A concrete implementation of Stopping Criterion must have '
        if not hasattr(self, 'abs_tol'):
            raise ParameterError(prefix + 'self.abs_tol (absolute tolerance)')
        if not hasattr(self, 'rel_tol'):
            raise ParameterError(prefix + 'self.rel_tol (relative tolerance)')
        if not hasattr(self, 'n_max'):
            raise ParameterError(prefix + 'self.n_max (maximum total samples)')
        if not hasattr(self, 'alpha'):
            raise ParameterError(prefix + 'self.alpha (uncertainty level)')
        if not hasattr(self, 'inflate'):
            raise ParameterError(prefix + 'self.inflate (inflation factor)')
        if not hasattr(self, 'stage'):
            raise ParameterError(prefix + 'self.stage (stage of the computation)')

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
        super_attributes = ['abs_tol', 'rel_tol', 'n_max', 'inflate', 'alpha']
        return univ_repr(self, "Stopping Criterion", super_attributes+attributes)
