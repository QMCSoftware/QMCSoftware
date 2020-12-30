from ..integrand._integrand import Integrand
from ..util import DistributionCompatibilityError, ParameterError, \
                   MethodImplementationError, _univ_repr


class StoppingCriterion(object):
    """ Stopping Criterion abstract class. DO NOT INSTANTIATE. """
    
    def __init__(self, allowed_levels, allowed_distribs):
        """
        Args:
            distribution (DiscreteDistribution): a DiscreteDistribution
            allowed_levels (list): which integrand types are supported: 'single', 'fixed-multi', 'adaptive-multi'
            allowed_distribs (list): list of names (strings) of compatible distributions
        """
        sname = type(self).__name__
        prefix = 'A concrete implementation of StoppingCriterion must have '
        # integrand check
        if (not hasattr(self, 'integrand')) or \
            (not isinstance(self.integrand,Integrand)):
            raise ParameterError(prefix + 'self.integrand, an Integrand instance')
        # true measure check
        if (not hasattr(self, 'true_measure')) or (self.true_measure!=self.integrand.true_measure):
            raise ParameterError(prefix + 'self.true_measure=self.integrand.true_measure')
        # discrete distribution check
        if (not hasattr(self, 'discrete_distrib')) or (self.discrete_distrib!=self.integrand.discrete_distrib):
            raise ParameterError(prefix + 'self.discrete_distrib=self.integrand.discrete_distrib')
        if type(self.integrand.discrete_distrib).__name__ not in allowed_distribs:
            raise ParameterError('%s must have a DiscreteDistribution in %s'%(sname,str(allowed_distribs)))
        # multilevel compatibility check
        if self.integrand.leveltype not in allowed_levels:
            raise ParameterError('Integrand is %s level but %s only supports %s level problems.' % \
            (self.integrand.leveltype,sname,allowed_levels))
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
