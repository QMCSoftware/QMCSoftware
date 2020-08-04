from ..discrete_distribution._discrete_distribution import DiscreteDistribution
from ..util import DistributionCompatibilityError, ParameterError, \
                   MethodImplementationError, _univ_repr


class StoppingCriterion(object):
    """ Stopping Criterion abstract class. DO NOT INSTANTIATE. """
    
    def __init__(self, distribution, integrand, allowed_levels, allowed_distribs):
        """
        Args:
            distribution (DiscreteDistribution): a DiscreteDistribution
            allowed_levels (list): which integrand types are supported: 'single', 'fixed-multi', 'adaptive-multi'
            allowed_distribs (list): list of names (strings) of compatible distributions
        """
        # check distribution compatibility with stopping_criterion
        s_name = type(self).__name__
        d_name = type(distribution).__name__
        if d_name not in allowed_distribs:
            error_message = "%s only accepts distributions: %s" %(s_name, str(allowed_distribs))
            raise DistributionCompatibilityError(error_message)
        # multilevel compatibility check
        if integrand.leveltype  not in allowed_levels:
            raise ParameterError('Integrand is %s level but %s only supports %s level problems.' % \
            (integrand.leveltype, s_name, allowed_levels))
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
    
    def set_tolerance(self, *args, **kwargs):
        """ ABSTRACT METHOD to reset the absolute tolerance. """

    def __repr__(self):
        return _univ_repr(self, "StoppingCriterion", self.parameters)
    
    def plot(self, *args, **kwargs):
        """ Create a plot relevant to the stopping criterion object. """
        raise MethodImplementationError(self,'plot')
