from ..integrand.abstract_integrand import AbstractIntegrand
from ..util import DistributionCompatibilityError, ParameterError, MethodImplementationError, _univ_repr
import numpy as np

# Diagnostic flag and utility for all stopping criteria
IS_PRINT_DIAGNOSTIC = False

def print_diagnostic(label, data):
    """
    Print diagnostic information for integration state.
    Args:
        label (str): Label for the diagnostic print (e.g., '[RESUME] Before resuming:').
        data (object): Data object containing integration state.
    """
    solution = getattr(data, 'solution', None)
    m = getattr(data, 'm', None)
    n_total = getattr(data, 'n_total', None)
    n_min = getattr(data, 'n_min', None)
    xfull = getattr(data, 'xfull', None)
    
    # Safely handle solution display
    try:
        solution_display = solution[0] if solution is not None and hasattr(solution, '__len__') and len(solution) > 0 else solution
    except (TypeError, IndexError):
        solution_display = solution
    
    # For convergence check steps, if solution is NaN, try to show current best estimate
    if (solution_display is not None and isinstance(solution_display, float) and 
        solution_display != solution_display and  # Check for NaN
        'After convergence check' in label):
        # Try different fallback estimates in order of preference
        best_estimate = None
        
        # Try muhat first
        muhat = getattr(data, 'muhat', None)
        if muhat is not None:
            try:
                muhat_display = muhat[0] if hasattr(muhat, '__len__') and len(muhat) > 0 else muhat
                if muhat_display is not None and not (isinstance(muhat_display, float) and muhat_display != muhat_display):
                    best_estimate = muhat_display
            except (TypeError, IndexError):
                pass
        
        # If muhat doesn't work, try computing from yfull
        if best_estimate is None:
            yfull = getattr(data, 'yfull', None)
            if yfull is not None and hasattr(yfull, 'size') and yfull.size > 0:
                try:
                    # Compute mean of finite values
                    finite_values = yfull[np.isfinite(yfull)]
                    if len(finite_values) > 0:
                        best_estimate = np.mean(finite_values)
                except (TypeError, IndexError, AttributeError):
                    pass
        
        # Use the best estimate if we found one
        if best_estimate is not None:
            solution_display = best_estimate
    
    # Safely handle m display    
    try:
        m_display = int(m[0]) if m is not None and hasattr(m, '__len__') and len(m) > 0 else (int(m) if m is not None else None)
    except (TypeError, IndexError):
        m_display = int(m) if m is not None else None
    n_total_display = int(n_total) if n_total is not None else None
    xfull_shape = getattr(xfull, 'shape', None)
    n_total_formatted = f"{n_total_display:>10}" if n_total_display is not None else f"{'None':>10}"
    # Safely handle solution display for formatting
    if solution_display is not None:
        if isinstance(solution_display, (int, float)):
            # Scalar case
            if np.isnan(solution_display):
                solution_formatted = f"{'nan':>10}"
            else:
                solution_formatted = f"{solution_display:>10.7f}"
        else:
            # Array case - convert to scalar if possible, otherwise show summary
            try:
                if hasattr(solution_display, 'shape') and solution_display.shape == ():
                    # 0-dimensional array
                    scalar_val = solution_display.item()
                    solution_formatted = f"{scalar_val:>10.7f}" if not np.isnan(scalar_val) else f"{'nan':>10}"
                elif hasattr(solution_display, '__len__') and len(solution_display) == 1:
                    # Single element array
                    scalar_val = solution_display[0]
                    solution_formatted = f"{scalar_val:>10.7f}" if not np.isnan(scalar_val) else f"{'nan':>10}"
                else:
                    # Multi-element array - show summary
                    solution_formatted = f"{'[array]':>10}"
            except (TypeError, IndexError, AttributeError):
                solution_formatted = f"{'[complex]':>10}"
    else:
        solution_formatted = f"{'None':>10}"
    m_formatted = f"{m_display:>4}" if m_display is not None else f"{'None':>4}"
    n_min_formatted = f"{n_min:>10}" if n_min is not None else f"{'None':>10}"
    print(f"{label}: solution: {solution_formatted}, n_total: {n_total_formatted}, n_min: {n_min_formatted}, m: {m_formatted}, xfull.shape: {xfull_shape}")

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
            
    def integrate(self, resume=None):
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
