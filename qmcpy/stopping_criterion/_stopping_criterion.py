from ..integrand.abstract_integrand import AbstractIntegrand
from ..util import DistributionCompatibilityError, ParameterError, MethodImplementationError, _univ_repr
import numpy as np

class StoppingCriterion(object):
    """ Stopping Criterion abstract class. DO NOT INSTANTIATE. """
    
    def __init__(self, allowed_levels, allowed_distribs, allow_vectorized_integrals):
        """
        Args:
            distribution (AbstractDiscreteDistribution): an AbstractDiscreteDistribution
            allowed_levels (list): which integrand types are supported: 'single', 'fixed-multi', 'adaptive-multi'
            allowed_distribs (list): list of compatible AbstractDiscreteDistribution classes
        """
        sname = type(self).__name__
        prefix = 'A concrete implementation of StoppingCriterion must have '
        # integrand check
        if (not hasattr(self, 'integrand')) or \
            (not isinstance(self.integrand,AbstractIntegrand)):
            raise ParameterError(prefix + 'self.integrand, an AbstractIntegrand instance')
        # true measure check
        if (not hasattr(self, 'true_measure')) or (self.true_measure!=self.integrand.true_measure):
            raise ParameterError(prefix + 'self.true_measure=self.integrand.true_measure')
        # discrete distribution check
        if (not hasattr(self, 'discrete_distrib')) or (self.discrete_distrib!=self.integrand.discrete_distrib):
            raise ParameterError(prefix + 'self.discrete_distrib=self.integrand.discrete_distrib')
        if not isinstance(self.discrete_distrib,tuple(allowed_distribs)):
            raise DistributionCompatibilityError('%s must have an AbstractDiscreteDistribution in %s'%(sname,str(allowed_distribs)))
        # multilevel compatibility check
        if self.integrand.leveltype not in allowed_levels:
            raise ParameterError('AbstractIntegrand is %s level but %s only supports %s level problems.'%(self.integrand.leveltype,sname,allowed_levels))
        if (not allow_vectorized_integrals) and self.integrand.d_indv!=(1,):
            raise ParameterError('Vectorized integrals (with d_indv>1 outputs per sample) are not supported by this stopping criterion')
        # parameter checks
        if not hasattr(self,'parameters'):
            self.parameters = []
            
    def integrate(self):
        """
        ABSTRACT METHOD to determine the number of samples needed to satisfy the tolerance.

        Returns:
            tuple: tuple containing:
                - solution (float): approximation to the integral
                - data (AccumulateData): an AccumulateData object
        """
        raise MethodImplementationError(self, 'integrate')
    
    def set_tolerance(self, *args, **kwargs):
        """ ABSTRACT METHOD to reset the absolute tolerance. """
        raise ParameterError("The %s StoppingCriterion does not yet support resetting tolerances.")

    def _compute_indv_alphas(self, alphas_comb):
        """
        Compute individual uncertainty levels required to achieve combined uncertainty levels. 

        Args:
            alphas_comb (np.ndarray): desired uncertainty levels on combined solutions. 
        
        Returns:
            np.ndarray: uncertainty levels on individual solutions"""
        alphas_indv = np.tile(1,self.integrand.d_indv)
        identity_dependency = True
        for k in np.ndindex(self.integrand.d_comb):
            comb_flags = np.tile(True,self.integrand.d_comb)
            comb_flags[k] = False
            flags_indv = self.integrand.dependency(comb_flags)
            if self.integrand.d_indv!=self.integrand.d_comb or (flags_indv!=comb_flags).any(): identity_dependency=False
            dependents_k = ~flags_indv
            n_dep_k = dependents_k.sum()
            alpha_k = alphas_comb[k]/n_dep_k
            alpha_k_mat = alpha_k*dependents_k
            alpha_k_mat[alpha_k_mat==0] = 1
            alphas_indv = np.minimum(alphas_indv,alpha_k_mat)
        return alphas_indv,identity_dependency
    def __repr__(self):
        return _univ_repr(self, "StoppingCriterion", self.parameters)
