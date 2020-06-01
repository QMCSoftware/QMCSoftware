from ..discrete_distribution._discrete_distribution import DiscreteDistribution
from ..util import DistributionCompatibilityError, ParameterError, \
                   MethodImplementationError, univ_repr


class StoppingCriterion(object):
    """ Stopping Criterion abstract class. DO NOT INSTANTIATE. """
    
    def __init__(self, distribution, allowed_levels, allowed_distribs):
        """
        Args:
            distribution (DiscreteDistribution): a DiscreteDistribution
            allowed_levels (str): supports 'single' or 'multi' level problems?
            allowed_distribs (list): list of names (strings) of compatible distributions
        """
        # check distribution compatibility with stopping_criterion
        s_name = type(self).__name__
        d_name = type(distribution).__name__
        if d_name not in allowed_distribs:
            error_message = "%s only accepts distributions: %s" %(s_name, str(allowed_distribs))
            raise DistributionCompatibilityError(error_message)
        # parameter checks
        if not hasattr(self,'parameters'):
            self.parameters = []
            
    def integrate(self):
        """
        ABSTRACT METHOD to determine the number of samples needed to satisfy the tolerance.

        Return:
            tuple: tuple containing:
                - solution (float): approximation to the integral
                - data (AccumulateData): an AccumulateData object
        """
        raise MethodImplementationError(self, 'integrate')

    def __repr__(self):
        return univ_repr(self, "StoppingCriterion", self.parameters)
