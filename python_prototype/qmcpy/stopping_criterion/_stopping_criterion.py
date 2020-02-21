""" Definition for abstract class StoppingCriterion """

from ..discrete_distribution._discrete_distribution import DiscreteDistribution
from ..util import DistributionCompatibilityError, ParameterError, \
                   MethodImplementationError, univ_repr


class StoppingCriterion(object):
    """
    Decide when to stopping_criterion

    Attributes:
        abs_tol: absolute error tolerance
        rel_tol: relative error tolerance
        n_max: maximum number of samples
        stage: stage of the computation
    """
    
    def __init__(self, distributions, allowed_distribs):
        """
        Args:
            distributions (DiscreteDistribution or list of Distributions): an instance of DiscreteDistribution
            allowed_distribs: distribution's compatible with the StoppingCriterion
        """
        if isinstance(distributions,DiscreteDistribution):
            # single level problem -> make it appear multi-level
            distributions = [distributions]
        for distribution in distributions:
            if type(distribution).__name__ not in allowed_distribs:
                error_message = "%s only accepts distributions: %s" %\
                    (type(self).__name__, str(allowed_distribs))
                raise DistributionCompatibilityError(error_message)
        prefix = 'A concrete implementation of Stopping Criterion must have '
        if not hasattr(self, 'abs_tol'):
            raise ParameterError(prefix + 'self.abs_tol (absolute tolerance)')
        if not hasattr(self, 'rel_tol'):
            raise ParameterError(prefix + 'self.rel_tol (relative tolerance)')
        if not hasattr(self, 'n_max'):
            raise ParameterError(prefix + 'self.n_max (maximum total samples)')
        if not hasattr(self, 'stage'):
            raise ParameterError(prefix + 'self.stage (stage of the computation)')
        if not hasattr(self,'parameters'):
            self.parameters = []
            
    def stop_yet(self):
        """ ABSTRACT METHOD
        Determine the number of samples needed to satisfy tolerance
        """
        raise MethodImplementationError(self, 'stop_yet')

    def __repr__(self):
        return univ_repr(self, "Stopping Criterion", self.parameters)
