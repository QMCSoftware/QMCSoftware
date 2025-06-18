from ..integrand.abstract_integrand import AbstractIntegrand
from ..util import DistributionCompatibilityError, ParameterError, MethodImplementationError, _univ_repr
import numpy as np


class AbstractStoppingCriterion(object):
    
    def __init__(self, allowed_distribs, allow_vectorized_integrals):
        """
        Args:
            allowed_distribs (list): Allowed discrete distribution classes.
            allow_vectorized_integrals (bool): Whether or not to allow integrands with vectorized outputs, 
                i.e., those with `integrand.d_indv!=()`. 
        """
        sname = type(self).__name__
        prefix = 'A concrete implementation of StoppingCriterion must have '
        # integrand check
        if (not hasattr(self, 'integrand')) or (not isinstance(self.integrand,AbstractIntegrand)):
            raise ParameterError(prefix + 'self.integrand, an Integrand instance')
        # true measure check
        if (not hasattr(self, 'true_measure')) or (self.true_measure!=self.integrand.true_measure):
            raise ParameterError(prefix + 'self.true_measure=self.integrand.true_measure')
        # discrete distribution check
        if (not hasattr(self, 'discrete_distrib')) or (self.discrete_distrib!=self.integrand.discrete_distrib):
            raise ParameterError(prefix + 'self.discrete_distrib=self.integrand.discrete_distrib')
        if not isinstance(self.discrete_distrib,tuple(allowed_distribs)):
            raise DistributionCompatibilityError('%s must have a DiscreteDistribution in %s'%(sname,str(allowed_distribs)))
        if (not allow_vectorized_integrals) and len(self.integrand.d_indv)>0:
            raise ParameterError('Vectorized integrals (with d_indv!=() outputs per sample) are not supported by this stopping criterion')
        # parameter checks
        if not hasattr(self,'parameters'):
            self.parameters = []
            
    def integrate(self):
        """
        *Abstract method* to determine the number of samples needed to satisfy the tolerance(s).

        Returns:
            solution (Union[float,np.ndarray]): Approximation to the integral with shape `integrand.d_comb`.
            data (Data): A data object.
        """
        raise MethodImplementationError(self, 'integrate')
    
    def _compute_indv_alphas(self, alphas_comb):
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
            alphas_indv = np.where(alpha_k_mat==0,alphas_indv,np.minimum(alpha_k_mat,alphas_indv))
        return alphas_indv,identity_dependency
    
    def set_tolerance(self, abs_tol=None, rel_tol=None, rmse_tol=None):
        """
        Reset the tolerances.
        
        Args:
            abs_tol (float): Absolute tolerance (if applicable). Reset if supplied, ignored otherwise. 
            rel_tol (float): Relative tolerance (if applicable). Reset if supplied, ignored otherwise. 
            rmse_tol (float): RMSE tolerance (if applicable). Reset if supplied, ignored if not. 
                If `rmse_tol` is not supplied but `abs_tol` is, then `rmse_tol = abs_tol / norm.ppf(1-alpha/2)`. 
        """
        raise MethodImplementationError(self, 'integrate')

    def __repr__(self):
        return _univ_repr(self, "AbstractStoppingCriterion", self.parameters)
