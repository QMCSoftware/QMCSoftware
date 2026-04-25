import copy

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


class _IterationTraceLogger(object):

    def __init__(self, stopping_criterion):
        self.enabled = bool(getattr(stopping_criterion, "trace_iterations", False))
        self.label = str(getattr(stopping_criterion, "trace_label", ""))
        self.throttle_iterations = bool(
            getattr(stopping_criterion, "trace_throttle_iterations", True)
        )
        self.header_printed = False
        self.table_header_printed = False
        self.iter_count = 0
        self.visible_columns = None
        self.pending_resume_signature = None

    @staticmethod
    def _state_signature(data):
        xfull = getattr(data, "xfull", None)
        return (
            getattr(data, "n_min", None),
            getattr(data, "n_total", None),
            getattr(data, "m", None),
            getattr(xfull, "shape", None),
        )

    def _print_header_once(self):
        if self.enabled and (not self.header_printed) and self.label:
            print(f"=== {self.label} iteration log ===")
            self.header_printed = True

    def _get_visible_columns(self, data):
        if self.visible_columns is not None:
            return self.visible_columns
        visible_columns = ["stage", "iter", "solution"]
        if getattr(data, "n_min", None) is not None:
            visible_columns.append("n_min")
        visible_columns.append("n_total")
        if getattr(data, "m", None) is not None:
            visible_columns.append("m")
        xfull = getattr(data, "xfull", None)
        if getattr(xfull, "shape", None) is not None:
            visible_columns.append("xfull.shape")
        self.visible_columns = tuple(visible_columns)
        return self.visible_columns

    def emit(self, stage, data, step_value=None, increment=False, iter_value=None):
        if not self.enabled:
            return
        self._print_header_once()
        if increment:
            self.iter_count += 1
            data._iter_count = self.iter_count
        elif iter_value is not None:
            data._iter_count = int(iter_value)
        else:
            data._iter_count = None
        if step_value is not None:
            data.m = int(step_value)
        visible_columns = self._get_visible_columns(data)
        print_diagnostic(
            stage,
            data,
            table_header=not self.table_header_printed,
            throttle_iterations=self.throttle_iterations,
            visible_columns=visible_columns,
        )
        self.table_header_printed = True

    def resume(self, data, step_value=None):
        previous_iter_count = getattr(data, "_iter_count", None)
        if previous_iter_count is not None:
            try:
                self.iter_count = int(previous_iter_count)
            except (TypeError, ValueError):
                pass
        self.emit(
            "RESUME",
            data,
            step_value=step_value,
            increment=False,
            iter_value=self.iter_count if self.iter_count > 0 else None,
        )
        self.pending_resume_signature = self._state_signature(data)

    def iteration(self, data, step_value=None):
        current_signature = self._state_signature(data)
        if (
            self.pending_resume_signature is not None
            and current_signature == self.pending_resume_signature
        ):
            self.pending_resume_signature = None
            return
        self.pending_resume_signature = None
        self.emit("ITER", data, step_value=step_value, increment=True)


def print_diagnostic(
    label,
    data,
    table_header=False,
    throttle_iterations=True,
    visible_columns=None,
):
    """Print diagnostic information for an integration state.

    Args:
        label (str): Stage label shown in the first column.
        data (object): Integration state carrying fields such as ``solution``,
            ``n_total``, ``n_min``, ``m``, and ``xfull``.
        table_header (bool, optional): Whether to print the compact table
            header before the row. Defaults to False.
        throttle_iterations (bool, optional): Whether to apply the current
            iteration-log throttling rules for ``ITER`` rows. Defaults to True.
            If False, every iteration row is printed.
        visible_columns (tuple[str, ...] | list[str] | None, optional): Ordered
            columns to print. Defaults to all supported columns.
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
    iter_count = getattr(data, "_iter_count", None)
    m_display = int(safe_get_first_element(m)) if m is not None else None
    iter_display = int(iter_count) if iter_count is not None else None
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
    iter_formatted = f"{iter_display:>4}" if iter_display is not None else " " * 4
    m_formatted = f"{m_display:>4}" if m_display is not None else f"{'None':>4}"

    throttle = iter_display
    if throttle_iterations and label == "ITER" and throttle is not None:
        if throttle > 1000 and throttle % 100 != 0:
            return
        if 10 < throttle <= 1000 and throttle % 10 != 0:
            return
    if visible_columns is None:
        visible_columns = (
            "stage",
            "iter",
            "solution",
            "n_min",
            "n_total",
            "m",
            "xfull.shape",
        )
    header_values = {
        "stage": f"{'stage':<12}",
        "iter": f"{'iter':>4}",
        "solution": f"{'solution':>10}",
        "n_min": f"{'n_min':>10}",
        "n_total": f"{'n_total':>10}",
        "m": f"{'m':>4}",
        "xfull.shape": f"{'xfull.shape':>16}",
    }
    row_values = {
        "stage": f"{label:<12}",
        "iter": iter_formatted,
        "solution": solution_formatted,
        "n_min": n_min_formatted,
        "n_total": n_total_formatted,
        "m": m_formatted,
        "xfull.shape": f"{str(xfull_shape):>16}",
    }
    if table_header:
        header_line = " ".join(header_values[column] for column in visible_columns)
        print(header_line)
        print("-" * len(header_line))
    print(" ".join(row_values[column] for column in visible_columns))


class AbstractStoppingCriterion(object):

    def __init__(self, allowed_distribs, allow_vectorized_integrals):
        """Initialize a stopping criterion base class.

        Args:
            allowed_distribs (list): Allowed discrete distribution classes.
            allow_vectorized_integrals (bool): Whether to allow integrands with
                vectorized outputs, meaning ``integrand.d_indv != ()``.
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
        """Determine the samples needed to satisfy the target tolerance.

        Args:
            resume (Data, optional): Existing integration state to resume from,
                if supported. Defaults to None.

        Returns:
            tuple[Union[float, np.ndarray], Data]: Approximation to the integral
                with shape ``integrand.d_comb`` and the corresponding data
                object.
        """
        raise MethodImplementationError(self, "integrate")

    def _make_trace_logger(self):
        """Create an iteration trace logger for this stopping criterion.

        Returns:
            _IterationTraceLogger: Trace logger configured from the stopping
                criterion's optional trace attributes.
        """
        return _IterationTraceLogger(self)

    def _prepare_resume_data(self, resume, validate_resume, restore_resume):
        """Validate and restore a resume checkpoint before integration.

        Args:
            resume (Data or None): Resume checkpoint passed to ``integrate``.
            validate_resume (callable): Validator taking ``resume`` and raising
                on incompatible state.
            restore_resume (callable): Restorer taking ``resume`` and mutating
                the current stopping criterion into a compatible resumed state.

        Returns:
            Data or None: The validated resume checkpoint, or None when no
            checkpoint was supplied.
        """
        if resume is None:
            return None
        validate_resume(resume)
        restore_resume(resume)
        return resume

    def _resume_value_equal(self, current, saved):
        if isinstance(current, np.ndarray) or isinstance(saved, np.ndarray):
            try:
                return np.array_equal(
                    np.asarray(current), np.asarray(saved), equal_nan=True
                )
            except TypeError:
                return np.array_equal(np.asarray(current), np.asarray(saved))
        if isinstance(current, (list, tuple)) and isinstance(saved, (list, tuple)):
            return len(current) == len(saved) and all(
                self._resume_value_equal(curr_i, saved_i)
                for curr_i, saved_i in zip(current, saved)
            )
        if isinstance(current, dict) and isinstance(saved, dict):
            return current.keys() == saved.keys() and all(
                self._resume_value_equal(current[key], saved[key]) for key in current
            )
        if (
            type(current) is type(saved)
            and hasattr(current, "parameters")
            and hasattr(saved, "parameters")
        ):
            return str(current) == str(saved)
        try:
            is_equal = current == saved
        except Exception:
            is_equal = str(current) == str(saved)
        if isinstance(is_equal, np.ndarray):
            return bool(np.all(is_equal))
        return bool(is_equal)

    def _require_resume_attrs(self, data, attrs):
        missing = [attr for attr in attrs if not hasattr(data, attr)]
        if missing:
            raise ParameterError(
                "resume data missing required attribute(s): %s"
                % ", ".join(sorted(missing))
            )

    def _validate_resume_object(self, label, current, saved, attrs):
        if saved is None:
            raise ParameterError("resume data missing %s state." % label)
        if type(saved) is not type(current):
            raise ParameterError(
                "resume data has incompatible %s type %s; expected %s."
                % (label, type(saved).__name__, type(current).__name__)
            )
        missing = [
            attr for attr in attrs if not hasattr(current, attr) or not hasattr(saved, attr)
        ]
        if missing:
            raise ParameterError(
                "resume data missing %s attribute(s): %s"
                % (label, ", ".join(sorted(missing)))
            )
        for attr in attrs:
            if not self._resume_value_equal(getattr(current, attr), getattr(saved, attr)):
                raise ParameterError(
                    "resume data has incompatible %s.%s." % (label, attr)
                )

    def _validate_resume_data(self, data, required_fields=()):
        self._require_resume_attrs(data, ("stopping_crit", "integrand", "true_measure", "discrete_distrib", "n_total") + tuple(required_fields))
        if type(data.stopping_crit) is not type(self):
            raise ParameterError("resume data was generated by %s, not %s." % (type(data.stopping_crit).__name__, type(self).__name__))
        self._validate_resume_object("integrand", self.integrand, data.integrand, ("d", "d_indv", "d_comb") + tuple(getattr(self.integrand, "parameters", [])))
        self._validate_resume_object("true_measure", self.true_measure, data.true_measure, ("d", "domain", "range") + tuple(getattr(self.true_measure, "parameters", [])))
        self._validate_resume_object(
            "discrete_distrib",
            self.discrete_distrib,
            data.discrete_distrib,
            ("d", "replications", "mimics", "entropy", "spawn_key")
            + tuple(getattr(self.discrete_distrib, "parameters", [])),
        )
        if int(data.n_total) > int(self.n_limit):
            raise ParameterError("resume data n_total=%d exceeds current n_limit=%d." % (int(data.n_total), int(self.n_limit)))

    def _validate_resume_with_state(self, data, required_fields=(), state_fields=()):
        self._validate_resume_data(data, required_fields=required_fields)
        self._require_resume_attrs(data, state_fields)
        if hasattr(self, "n_init") and int(data.n_total) < int(self.n_init):
            raise ParameterError("resume data must include at least n_init samples.")

    def _restore_resume_rng_state(self, data):
        saved_distrib = data.discrete_distrib
        if not hasattr(saved_distrib, "rng") or not hasattr(self.discrete_distrib, "rng"):
            raise ParameterError("resume data is missing discrete distribution RNG state.")
        self.discrete_distrib.rng.bit_generator.state = copy.deepcopy(
            saved_distrib.rng.bit_generator.state
        )

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
            alphas_indv = np.where(alpha_k_mat == 0, alphas_indv, np.minimum(alpha_k_mat, alphas_indv))
        return alphas_indv, identity_dependency

    def set_tolerance(self, abs_tol=None, rel_tol=None, rmse_tol=None):
        """Reset the tolerances.

        Args:
            abs_tol (float): Absolute tolerance, when supported. If supplied,
                reset it; otherwise ignore it.
            rel_tol (float): Relative tolerance, when supported. If supplied,
                reset it; otherwise ignore it.
            rmse_tol (float): RMSE tolerance, when supported. If supplied,
                reset it; otherwise ignore it. If ``rmse_tol`` is not supplied
                but ``abs_tol`` is, then ``rmse_tol = abs_tol / norm.ppf(1 -
                alpha / 2)``.
        """
        raise MethodImplementationError(self, "set_tolerance")

    def __repr__(self):
        return _univ_repr(self, "AbstractStoppingCriterion", self.parameters)
