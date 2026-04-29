"""Diagnostics helpers for stopping-criterion iteration tracing."""

import numpy as np
from math import log10

# ITER rows up to this count are always printed; above it the log-scale throttle applies.
_THROTTLE_FREE_ITER_THRESHOLD = 30

class _IterationTraceLogger(object):
    def __init__(self, stopping_criterion):
        """Create a trace logger bound to the given stopping criterion.

        Args:
            stopping_criterion: Stopping criterion instance. The logger reads
                the optional attributes ``trace_iterations`` (bool),
                ``trace_label`` (str), and ``trace_throttle_iterations`` (bool)
                to configure itself.
        """
        self.enabled = bool(getattr(stopping_criterion, "trace_iterations", False))
        self.label = str(getattr(stopping_criterion, "trace_label", ""))
        self.throttle_iterations = bool(getattr(stopping_criterion, "trace_throttle_iterations", True))
        self.store_throttled_iterations = bool(getattr(stopping_criterion, "trace_store_throttled_iterations", False))
        self.header_printed = False
        self.table_header_printed = False
        self.iter_count = 0
        self.visible_columns = None
        self.pending_resume_signature = None
        self._last_iter_snapshot = None
        self._last_iter_count = 0
        self._last_printed_iter_count = 0

    def _would_be_throttled(self, iter_count):
        """Return True if an ITER row with this count would be suppressed by throttling."""
        if not self.throttle_iterations:
            return False
        if iter_count is None or iter_count <= _THROTTLE_FREE_ITER_THRESHOLD:
            return False
        step = 10 ** int(log10(iter_count))
        return iter_count % step != 0

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
        if getattr(data, "rmse_estimate", None) is not None:
            visible_columns.append("rmse_estimate")
        if getattr(data, "rmse_tol", None) is not None:
            visible_columns.append("rmse_tol")
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
        if stage == "ITER":
            self._last_iter_snapshot = data
            self._last_iter_count = data._iter_count if data._iter_count is not None else 0
            if not self._would_be_throttled(data._iter_count):
                self._last_printed_iter_count = self._last_iter_count
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

    def _flush_last_if_suppressed(self):
        """Force-print the last ITER row if it was suppressed by throttling."""
        if not self.enabled:
            return
        if self._last_iter_snapshot is None:
            return
        if self._last_iter_count == self._last_printed_iter_count:
            return
        print_diagnostic(
            "ITER",
            self._last_iter_snapshot,
            table_header=False,
            throttle_iterations=False,
            visible_columns=self.visible_columns,
        )
        self._last_printed_iter_count = self._last_iter_count

    def finalize(self):
        """Force-print the last ITER row if throttling suppressed it, then clear snapshot."""
        self._flush_last_if_suppressed()
        self._last_iter_snapshot = None


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
    rmse_estimate = getattr(data, "rmse_estimate", None)
    rmse_tol = getattr(data, "rmse_tol", None)
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

    def _positive_float(value):
        value = safe_get_first_element(value)
        if value is None:
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(value) or value <= 0:
            return None
        return value

    sc = getattr(data, "stopping_crit", None)
    tolerance_candidates = []
    for tol_value in (
        getattr(data, "abs_tol", None),
        rmse_tol,
        getattr(sc, "abs_tol", None) if sc is not None else None,
        getattr(sc, "rmse_tol", None) if sc is not None else None,
    ):
        tol_float = _positive_float(tol_value)
        if tol_float is not None:
            tolerance_candidates.append(tol_float)

    solution_decimals = 7
    if tolerance_candidates:
        solution_decimals = int(np.ceil(-np.log10(min(tolerance_candidates)))) + 1
        solution_decimals = max(7, min(12, solution_decimals))
    solution_width = max(10, solution_decimals + 4)

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
        solution_formatted = f"{solution_display:>{solution_width}.{solution_decimals}f}"
    else:
        solution_formatted = f"{'nan':>{solution_width}}"
    n_min_formatted = f"{int(n_min):>10}" if n_min is not None else f"{'None':>10}"
    bound_diff_formatted = _format_bound(bound_diff_display)
    comb_bound_diff_formatted = _format_bound(comb_bound_diff_display)
    bound_half_width_formatted = _format_bound(bound_half_width_display)
    bias_estimate_formatted = _format_bound(bias_estimate_display)
    rmse_estimate_formatted = _format_bound(rmse_estimate)
    rmse_tol_formatted = _format_bound(rmse_tol)
    iter_formatted = f"{iter_display:>4}" if iter_display is not None else " " * 4
    m_formatted = f"{m_display:>6}" if m_display is not None else f"{'None':>6}"

    throttle = iter_display
    if throttle_iterations and label == "ITER" and throttle is not None and throttle > _THROTTLE_FREE_ITER_THRESHOLD:
        step = 10 ** int(log10(throttle))
        if throttle % step != 0:
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
        if rmse_estimate is not None:
            visible_columns.append("rmse_estimate")
        if rmse_tol is not None:
            visible_columns.append("rmse_tol")
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
        "solution": f"{'solution':>{solution_width}}",
        "bound_diff": f"{'bound_diff':>15}",
        "comb_bound_diff": f"{'comb_bound_diff':>15}",
        "bound_half_width": f"{'bound_half_width':>15}",
        "bias_estimate": f"{'bias_estimate':>15}",
        "rmse_estimate": f"{'rmse_estimate':>15}",
        "rmse_tol": f"{'rmse_tol':>15}",
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
        "rmse_estimate": rmse_estimate_formatted,
        "rmse_tol": rmse_tol_formatted,
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
