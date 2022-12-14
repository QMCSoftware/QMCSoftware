from ..integrand._integrand import Integrand
from ..util import DistributionCompatibilityError, ParameterError, MethodImplementationError, _univ_repr


class StoppingCriterion(object):
    """ Stopping Criterion abstract class. DO NOT INSTANTIATE. """
    
    def __init__(self, allowed_levels, allowed_distribs, allow_vectorized_integrals):
        """
        Args:
            distribution (DiscreteDistribution): a DiscreteDistribution
            allowed_levels (list): which integrand types are supported: 'single', 'fixed-multi', 'adaptive-multi'
            allowed_distribs (list): list of compatible DiscreteDistribution classes
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
        if not isinstance(self.discrete_distrib,tuple(allowed_distribs)):
            raise DistributionCompatibilityError('%s must have a DiscreteDistribution in %s'%(sname,str(allowed_distribs)))
        # multilevel compatibility check
        if self.integrand.leveltype not in allowed_levels:
            raise ParameterError('Integrand is %s level but %s only supports %s level problems.'%(self.integrand.leveltype,sname,allowed_levels))
        if (not allow_vectorized_integrals) and self.integrand.dprime!=(1,):
            raise ParameterError('Vectorized integrals (with dprime>1 outputs per sample) are not supported by this stopping criterion')
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
        raise ParameterError("The %s StoppingCriterioin does not yet support resetting tolerances.")

    def __repr__(self):
        return _univ_repr(self, "StoppingCriterion", self.parameters)
