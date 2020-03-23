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
    
    def __init__(self, distribution, allowed_levels, allowed_distribs):
        """
        Check StoppingCriterion parameters

        Args:
            distribution: DiscreteDistribution or list of DiscreteDistributions
            allowed_levels (string): stopping criteron works with 'single' or 'multi' level problems
            allowed_distribs (list): list of names (strings) of compatible distributions
        """
        # check compatable level and StoppingCriterion
        if isinstance(distribution,DiscreteDistribution):
            levels = 'single'
            distribution = [distribution] # make it apprea multi-level for next check
        else:
            levels = 'multi'
        if levels=='multi' and allowed_levels=='single':
            raise NotYetImplemented('''
                StoppingCriterion not implemented for multi-level problems.
                Use CLT stopping criterion with an iid distribution for multi-level problems.''')
        # check distribution compatibility with stopping_criterion
        s_name = type(self).__name__
        for distrib in distribution:
            d_name = type(distrib).__name__
            if d_name not in allowed_distribs:
                error_message = "%s only accepts distributions: %s" %(s_name, str(allowed_distribs))
                raise DistributionCompatibilityError(error_message)
        # parameter checks
        prefix = 'A concrete implementation of Stopping Criterion must have '
        if not hasattr(self, 'abs_tol'):
            raise ParameterError(prefix + 'self.abs_tol (absolute tolerance)')
        if not hasattr(self, 'rel_tol'):
            raise ParameterError(prefix + 'self.rel_tol (relative tolerance)')
        if not hasattr(self, 'n_max'):
            raise ParameterError(prefix + 'self.n_max (maximum total samples)')
        if not hasattr(self,'parameters'):
            self.parameters = []
            
    def integrate(self):
        """ ABSTRACT METHOD
        Determine the number of samples needed to satisfy tolerance

        Return:
            solution (float): approximate integral
            data (AccumulateData): an AccumulateData object
        """
        raise MethodImplementationError(self, 'integrate')

    def __repr__(self):
        return univ_repr(self, "StoppingCriterion", self.parameters)
