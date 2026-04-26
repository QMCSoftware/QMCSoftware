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
        """Create a trace logger bound to the given stopping criterion.

        Args:
            stopping_criterion: Stopping criterion instance.  The logger reads
                the optional attributes ``trace_iterations`` (bool),
                ``trace_label`` (str), and ``trace_throttle_iterations`` (bool)
                to configure itself.
        """
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
        """Return a hashable snapshot of the data fields used to detect duplicate rows.

        Args:
            data (object): Integration state object.

        Returns:
            tuple: ``(n_min, n_total, m, xfull.shape)``.
        """
        xfull = getattr(data, "xfull", None)
        return (
            getattr(data, "n_min", None),
            getattr(data, "n_total", None),
            getattr(data, "m", None),
            getattr(xfull, "shape", None),
        )

    def _print_header_once(self):
        """Print the section header line exactly once per logger instance."""
        if self.enabled and (not self.header_printed) and self.label:
            print(f"=== {self.label} iteration log ===")
            self.header_printed = True

    def _get_visible_columns(self, data):
        """Return the ordered list of column names to display, inferred from data.

        The result is cached after the first call so all rows share the same
        columns.

        Args:
            data (object): Integration state object used to determine which
                optional columns are present.

        Returns:
            tuple[str, ...]: Column names from the set ``{'stage', 'iter',
                'solution', 'bound_diff', 'comb_bound_diff',
                'bound_half_width', 'bias_estimate', 'n_min', 'n_total',
                'm', 'xfull.shape'}``.
        """
        if self.visible_columns is not None:
            return self.visible_columns
        visible_columns = ["stage", "iter", "solution"]
        if getattr(data, "bound_diff", None) is not None:
            visible_columns.append("bound_diff")
        if getattr(data, "comb_bound_diff", None) is not None:
            visible_columns.append("comb_bound_diff")
        if getattr(data, "bound_half_width", None) is not None:
            visible_columns.append("bound_half_width")
        if getattr(data, "bias_estimate", None) is not None:
            visible_columns.append("bias_estimate")
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
        """Print one diagnostic row for the given stage label.

        Args:
            stage (str): Row label, e.g. ``"ITER"`` or ``"RESUME"``.
            data (object): Integration state object.
            step_value (int | None, optional): Value to assign to ``data.m``
                before printing. Defaults to None.
            increment (bool, optional): If True, advance the internal iteration
                counter and assign the new value to ``data._iter_count``.
                Defaults to False.
            iter_value (int | None, optional): Explicit iteration count to
                display (overrides ``increment``). Defaults to None.
        """
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
        """Emit a RESUME row and snapshot the current state for duplicate suppression.

        Reads ``data._iter_count`` to restore the iteration counter so that the
        next :meth:`iteration` call continues counting from the right number.
        If the state does not change between the RESUME and the first ITER call
        (e.g. no new samples were needed), that first ITER row is suppressed.

        Args:
            data (object): Integration state object from the resume checkpoint.
            step_value (int | None, optional): Value to assign to ``data.m``
                before printing. Defaults to None.
        """
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
        """Emit an ITER row, unless state is unchanged since the last resume.

        If :meth:`resume` was just called and the data state has not changed
        (same ``n_total``, ``n_min``, ``m``, and ``xfull.shape``), the row is
        suppressed to avoid a duplicate log entry.

        Args:
            data (object): Current integration state object.
            step_value (int | None, optional): Value to assign to ``data.m``
                before printing. Defaults to None.
        """
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
    bound_diff = getattr(data, "bound_diff", None)
    comb_bound_diff = getattr(data, "comb_bound_diff", None)
    bound_half_width = getattr(data, "bound_half_width", None)
    bias_estimate = getattr(data, "bias_estimate", None)
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
    bound_diff_display = safe_get_first_element(bound_diff)
    comb_bound_diff_display = safe_get_first_element(comb_bound_diff)
    bound_half_width_display = safe_get_first_element(bound_half_width)
    bias_estimate_display = safe_get_first_element(bias_estimate)
    iter_count = getattr(data, "_iter_count", None)
    m_display = int(safe_get_first_element(m)) if m is not None else None
    iter_display = int(iter_count) if iter_count is not None else None
    n_total_display = int(n_total) if n_total is not None else None
    xfull_shape = getattr(xfull, "shape", None)

    def _format_bound(value):
        if value is None:
            return f"{'None':>15}"
        try:
            value = float(value)
        except (TypeError, ValueError):
            return f"{str(value):>15}"
        if np.isnan(value):
            return f"{'nan':>15}"
        return f"{value:>15.3e}"

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
    bound_diff_formatted = _format_bound(bound_diff_display)
    comb_bound_diff_formatted = _format_bound(comb_bound_diff_display)
    bound_half_width_formatted = _format_bound(bound_half_width_display)
    bias_estimate_formatted = _format_bound(bias_estimate_display)
    iter_formatted = f"{iter_display:>4}" if iter_display is not None else " " * 4
    m_formatted = f"{m_display:>6}" if m_display is not None else f"{'None':>6}"

    throttle = iter_display
    if throttle_iterations and label == "ITER" and throttle is not None:
        if throttle > 1000 and throttle % 100 != 0:
            return
        if 10 < throttle <= 1000 and throttle % 10 != 0:
            return
    if visible_columns is None:
        visible_columns = ["stage", "iter", "solution"]
        if bound_diff is not None:
            visible_columns.append("bound_diff")
        if comb_bound_diff is not None:
            visible_columns.append("comb_bound_diff")
        if bound_half_width is not None:
            visible_columns.append("bound_half_width")
        if bias_estimate is not None:
            visible_columns.append("bias_estimate")
        visible_columns.append("n_min")
        visible_columns.append("n_total")
        if m is not None:
            visible_columns.append("m")
        if xfull_shape is not None:
            visible_columns.append("xfull.shape")
        visible_columns = tuple(visible_columns)
    header_values = {
        "stage": f"{'stage':<12}",
        "iter": f"{'iter':>4}",
        "solution": f"{'solution':>10}",
        "bound_diff": f"{'bound_diff':>15}",
        "comb_bound_diff": f"{'comb_bound_diff':>15}",
        "bound_half_width": f"{'bound_half_width':>15}",
        "bias_estimate": f"{'bias_estimate':>15}",
        "n_min": f"{'n_min':>10}",
        "n_total": f"{'n_total':>10}",
        "m": f"{'m':>6}",
        "xfull.shape": f"{'xfull.shape':>16}",
    }
    row_values = {
        "stage": f"{label:<12}",
        "iter": iter_formatted,
        "solution": solution_formatted,
        "bound_diff": bound_diff_formatted,
        "comb_bound_diff": comb_bound_diff_formatted,
        "bound_half_width": bound_half_width_formatted,
        "bias_estimate": bias_estimate_formatted,
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
        """Deep equality check tolerant of arrays, lists, dicts, and QMCPy objects.

        Args:
            current: Value from the live stopping criterion.
            saved: Value from the resume checkpoint.

        Returns:
            bool: True when the two values are considered equal.
        """
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
        """Raise ParameterError if any attribute in *attrs* is absent from *data*.

        Args:
            data (Data): Resume checkpoint.
            attrs (tuple[str, ...]): Required attribute names.

        Raises:
            ParameterError: If one or more attributes are missing.
        """
        missing = [attr for attr in attrs if not hasattr(data, attr)]
        if missing:
            raise ParameterError(
                "resume data missing required attribute(s): %s"
                % ", ".join(sorted(missing))
            )

    def _validate_resume_object(self, label, current, saved, attrs):
        """Validate that a saved sub-object is compatible with the current one.

        Checks type equality and then compares each attribute listed in *attrs*
        using :meth:`_resume_value_equal`.

        Args:
            label (str): Human-readable name used in error messages.
            current: Live object (integrand, true measure, etc.).
            saved: Saved object from the resume checkpoint.
            attrs (tuple[str, ...]): Attribute names to compare.

        Raises:
            ParameterError: If the objects are incompatible.
        """
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
        """Run standard cross-cutting resume compatibility checks.

        Validates stopping criterion type, integrand, true measure, discrete
        distribution, and ``n_total`` against ``n_limit``.

        Args:
            data (Data): Resume checkpoint to validate.
            required_fields (tuple[str, ...], optional): Additional attribute
                names that must be present on *data*. Defaults to ``()``.

        Raises:
            ParameterError: If any compatibility check fails.
        """
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
        """Validate resume data including algorithm-specific state fields.

        Calls :meth:`_validate_resume_data` and additionally checks that all
        *state_fields* are present and that ``n_total >= n_init``.

        Args:
            data (Data): Resume checkpoint to validate.
            required_fields (tuple[str, ...], optional): Extra data attributes
                required beyond the standard set. Defaults to ``()``.
            state_fields (tuple[str, ...], optional): Algorithm-state attributes
                that must also be present. Defaults to ``()``.

        Raises:
            ParameterError: If any compatibility check fails.
        """
        self._validate_resume_data(data, required_fields=required_fields)
        self._require_resume_attrs(data, state_fields)
        if hasattr(self, "n_init") and int(data.n_total) < int(self.n_init):
            raise ParameterError("resume data must include at least n_init samples.")

    def _restore_resume_rng_state(self, data):
        """Deep-copy the saved RNG state into the live discrete distribution.

        Ensures that samples drawn after resuming are independent of those
        already stored in the checkpoint.

        Args:
            data (Data): Resume checkpoint carrying the saved distribution in
                ``data.discrete_distrib``.

        Raises:
            ParameterError: If either distribution lacks an ``rng`` attribute.
        """
        saved_distrib = data.discrete_distrib
        if not hasattr(saved_distrib, "rng") or not hasattr(self.discrete_distrib, "rng"):
            raise ParameterError("resume data is missing discrete distribution RNG state.")
        self.discrete_distrib.rng.bit_generator.state = copy.deepcopy(
            saved_distrib.rng.bit_generator.state
        )

    def _compute_indv_alphas(self, alphas_comb):
        """Distribute combined confidence levels to individual integrand dimensions.

        Uses the integrand dependency map to allocate the per-combined-output
        alpha budget down to each individual output dimension.

        Args:
            alphas_comb (np.ndarray): Per-combined-output confidence levels with
                shape ``integrand.d_comb``.

        Returns:
            tuple[np.ndarray, bool]: ``(alphas_indv, identity_dependency)``
                where *alphas_indv* has shape ``integrand.d_indv`` and
                *identity_dependency* is True when each combined output depends
                on exactly its matching individual output.
        """
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
