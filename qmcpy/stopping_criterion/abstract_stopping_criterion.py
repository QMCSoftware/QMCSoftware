from ..integrand.abstract_integrand import AbstractIntegrand
from ..util import (
    DistributionCompatibilityError,
    ParameterError,
    MethodImplementationError,
    _univ_repr,
)
import numpy as np


# Optional diagnostic hook for resume-aware stopping criteria.
IS_PRINT_DIAGNOSTIC = False


def print_diagnostic(label, data, table_header=False):
    """Print diagnostic information for integration state.

    When ``table_header`` is True, this prints a compact table header before the row.
    """
    solution = getattr(data, "solution", None)
    m = getattr(data, "m", None)
    n_total = getattr(data, "n_total", None)
    n_min = getattr(data, "n_min", None)
    xfull = getattr(data, "xfull", None)

    def safe_get_first_element(obj):
        try:
            if isinstance(obj, (list, tuple, np.ndarray)) and obj is not None:
                if hasattr(obj, "shape") and obj.shape == ():
                    return obj
                if len(obj) > 0:
                    return obj[0]
            return obj
        except (TypeError, AttributeError):
            return obj

    solution_display = safe_get_first_element(solution)
    m_display = int(safe_get_first_element(m)) if m is not None else None
    n_total_display = int(n_total) if n_total is not None else None
    xfull_shape = getattr(xfull, "shape", None)

    n_total_formatted = (
        f"{n_total_display:>10}" if n_total_display is not None else f"{'None':>10}"
    )
    if solution_display is not None and not (
        isinstance(solution_display, float) and solution_display != solution_display
    ):
        solution_formatted = f"{solution_display:>10.7f}"
    else:
        solution_formatted = f"{'nan':>10}"
    n_min_formatted = f"{int(n_min):>10}" if n_min is not None else f"{'None':>10}"
    m_formatted = f"{m_display:>4}" if m_display is not None else f"{'None':>4}"

    if table_header:
        print(
            f"{'stage':<12} {'solution':>10} {'n_total':>10} {'n_min':>10} "
            f"{'m':>4} {'xfull.shape':>16}"
        )
        print("-" * 72)
    print(
        f"{label:<12} {solution_formatted} {n_total_formatted} "
        f"{n_min_formatted} {m_formatted} {str(xfull_shape):>16}"
    )


class AbstractStoppingCriterion(object):

    def __init__(self, allowed_distribs, allow_vectorized_integrals):
        """
        Args:
            allowed_distribs (list): Allowed discrete distribution classes.
            allow_vectorized_integrals (bool): Whether or not to allow integrands with vectorized outputs,
                i.e., those with `integrand.d_indv!=()`.
        """
        sname = type(self).__name__
        prefix = "A concrete implementation of StoppingCriterion must have "
        # integrand check
        if (not hasattr(self, "integrand")) or (
            not isinstance(self.integrand, AbstractIntegrand)
        ):
            raise ParameterError(prefix + "self.integrand, an Integrand instance")
        # true measure check
        if (not hasattr(self, "true_measure")) or (
            self.true_measure != self.integrand.true_measure
        ):
            raise ParameterError(
                prefix + "self.true_measure=self.integrand.true_measure"
            )
        # discrete distribution check
        if (not hasattr(self, "discrete_distrib")) or (
            self.discrete_distrib != self.integrand.discrete_distrib
        ):
            raise ParameterError(
                prefix + "self.discrete_distrib=self.integrand.discrete_distrib"
            )
        if not isinstance(self.discrete_distrib, tuple(allowed_distribs)):
            raise DistributionCompatibilityError(
                "%s must have a DiscreteDistribution in %s"
                % (sname, str(allowed_distribs))
            )
        if (not allow_vectorized_integrals) and len(self.integrand.d_indv) > 0:
            raise ParameterError(
                "Vectorized integrals (with d_indv!=() outputs per sample) are not supported by this stopping criterion"
            )
        # parameter checks
        if not hasattr(self, "parameters"):
            self.parameters = []

    def integrate(self, resume=None):
        """
        *Abstract method* to determine the number of samples needed to satisfy the tolerance(s).

        Args:
            resume (Data, optional): Existing integration state to resume from, if supported.

        Returns:
            solution (Union[float, np.ndarray]): Approximation to the integral with shape `integrand.d_comb`.
            data (Data): A data object.
        """
        raise MethodImplementationError(self, "integrate")

    def _compute_indv_alphas(self, alphas_comb):
        alphas_indv = np.tile(1, self.integrand.d_indv)
        identity_dependency = True
        for k in np.ndindex(self.integrand.d_comb):
            comb_flags = np.tile(True, self.integrand.d_comb)
            comb_flags[k] = False
            flags_indv = self.integrand.dependency(comb_flags)
            if (
                self.integrand.d_indv != self.integrand.d_comb
                or (flags_indv != comb_flags).any()
            ):
                identity_dependency = False
            dependents_k = ~flags_indv
            n_dep_k = dependents_k.sum()
            alpha_k = alphas_comb[k] / n_dep_k
            alpha_k_mat = alpha_k * dependents_k
            alphas_indv = np.where(
                alpha_k_mat == 0, alphas_indv, np.minimum(alpha_k_mat, alphas_indv)
            )
        return alphas_indv, identity_dependency

    def set_tolerance(self, abs_tol=None, rel_tol=None, rmse_tol=None):
        """
        Reset the tolerances.

        Args:
            abs_tol (float): Absolute tolerance (if applicable). Reset if supplied, ignored otherwise.
            rel_tol (float): Relative tolerance (if applicable). Reset if supplied, ignored otherwise.
            rmse_tol (float): RMSE tolerance (if applicable). Reset if supplied, ignored if not.
                If `rmse_tol` is not supplied but `abs_tol` is, then `rmse_tol = abs_tol / norm.ppf(1-alpha/2)`.
        """
        raise MethodImplementationError(self, "integrate")

    def __repr__(self):
        return _univ_repr(self, "AbstractStoppingCriterion", self.parameters)
