"""Diagnostics helpers for stopping-criterion iteration tracing."""

import io
import numpy as np
import sys
from contextlib import redirect_stdout
from math import log10

# ITER rows up to this count are always printed; above it the log-scale throttle applies.
_THROTTLE_ITER_THRESHOLD = 30

_ITERATION_HISTORY_COLUMNS = (
    "stage",
    "iter",
    "solution",
    "bound_diff",
    "comb_bound_diff",
    "bound_half_width",
    "bias_estimate",
    "rmse_estimate",
    "rmse_tol",
    "n_min",
    "n_total",
    "m",
    "xfull.shape",
    "printed",
)


class IterationHistoryTable(object):
    """Column-oriented storage for iteration trace rows."""

    __slots__ = ("_columns", "visible_columns", "context")

    def __init__(self):
        self._columns = {column: [] for column in _ITERATION_HISTORY_COLUMNS}
        self.visible_columns = None
        self.context = None

    @property
    def columns(self):
        return _ITERATION_HISTORY_COLUMNS

    @property
    def shape(self):
        return (len(self), len(self.columns))

    def __len__(self):
        return len(self._columns["stage"])

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._columns[key]
        return self.row(key)

    def __repr__(self):
        return "IterationHistoryTable(rows=%d, columns=%s)" % (
            len(self),
            self.columns,
        )

    def append(self, stage, row, visible_columns=None, printed=True):
        if self.visible_columns is None and visible_columns is not None:
            self.visible_columns = tuple(visible_columns)
        self._columns["stage"].append(stage)
        self._columns["iter"].append(row["iter"])
        self._columns["solution"].append(row["solution"])
        self._columns["bound_diff"].append(row["bound_diff"])
        self._columns["comb_bound_diff"].append(row["comb_bound_diff"])
        self._columns["bound_half_width"].append(row["bound_half_width"])
        self._columns["bias_estimate"].append(row["bias_estimate"])
        self._columns["rmse_estimate"].append(row["rmse_estimate"])
        self._columns["rmse_tol"].append(row["rmse_tol"])
        self._columns["n_min"].append(row["n_min"])
        self._columns["n_total"].append(row["n_total"])
        self._columns["m"].append(row["m"])
        self._columns["xfull.shape"].append(row["xfull.shape"])
        self._columns["printed"].append(bool(printed))
        return len(self) - 1

    def mark_printed(self, index, printed=True):
        self._columns["printed"][index] = bool(printed)

    def row(self, index):
        return {column: self._columns[column][index] for column in self.columns}

    def rows(self):
        return [self.row(index) for index in range(len(self))]

    def to_dict(self):
        return {column: list(values) for column, values in self._columns.items()}


def _safe_get_first_element(obj):
    try:
        if isinstance(obj, (list, tuple, np.ndarray)) and obj is not None:
            if hasattr(obj, "shape") and obj.shape == ():
                return obj
            if len(obj) > 0:
                return obj[0]
        return obj
    except (TypeError, AttributeError):
        return obj


def _positive_float(value):
    value = _safe_get_first_element(value)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value) or value <= 0:
        return None
    return value


def _extract_diagnostic_row(data):
    xfull = getattr(data, "xfull", None)
    m = getattr(data, "m", None)
    iter_count = getattr(data, "_iter_count", None)
    n_min = getattr(data, "n_min", None)
    n_total = getattr(data, "n_total", None)
    return {
        "iter": int(iter_count) if iter_count is not None else None,
        "solution": _safe_get_first_element(getattr(data, "solution", None)),
        "bound_diff": _safe_get_first_element(getattr(data, "bound_diff", None)),
        "comb_bound_diff": _safe_get_first_element(
            getattr(data, "comb_bound_diff", None)
        ),
        "bound_half_width": _safe_get_first_element(
            getattr(data, "bound_half_width", None)
        ),
        "bias_estimate": _safe_get_first_element(
            getattr(data, "bias_estimate", None)
        ),
        "rmse_estimate": _safe_get_first_element(
            getattr(data, "rmse_estimate", None)
        ),
        "rmse_tol": _safe_get_first_element(getattr(data, "rmse_tol", None)),
        "n_min": int(n_min) if n_min is not None else None,
        "n_total": int(n_total) if n_total is not None else None,
        "m": int(_safe_get_first_element(m)) if m is not None else None,
        "xfull.shape": getattr(xfull, "shape", None),
    }


def _visible_columns_from_row(row, include_n_min=False):
    visible_columns = ["stage", "iter", "solution"]
    if row["bound_diff"] is not None:
        visible_columns.append("bound_diff")
    if row["comb_bound_diff"] is not None:
        visible_columns.append("comb_bound_diff")
    if row["bound_half_width"] is not None:
        visible_columns.append("bound_half_width")
    if row["bias_estimate"] is not None:
        visible_columns.append("bias_estimate")
    if row["rmse_estimate"] is not None:
        visible_columns.append("rmse_estimate")
    if row["rmse_tol"] is not None:
        visible_columns.append("rmse_tol")
    if include_n_min or row["n_min"] is not None:
        visible_columns.append("n_min")
    visible_columns.append("n_total")
    if row["m"] is not None:
        visible_columns.append("m")
    if row["xfull.shape"] is not None:
        visible_columns.append("xfull.shape")
    return tuple(visible_columns)


def _history_context_from_stopping_criterion(stopping_criterion):
    return {
        "trace_label": str(getattr(stopping_criterion, "trace_label", "")),
        "abs_tol": getattr(stopping_criterion, "abs_tol", None),
        "rel_tol": getattr(stopping_criterion, "rel_tol", None),
        "rmse_tol": getattr(stopping_criterion, "rmse_tol", None),
        "target_rmse_tol": getattr(stopping_criterion, "target_rmse_tol", None),
    }


def _build_history_stopping_criterion(history, stopping_criterion=None):
    context = dict(getattr(history, "context", None) or {})
    if not context and stopping_criterion is not None:
        context = _history_context_from_stopping_criterion(stopping_criterion)
    return type("HistoryStoppingCriterion", (), context)()


def _build_history_data(history, index, stopping_criterion=None):
    row = history.row(index)
    context = dict(getattr(history, "context", None) or {})
    data = type("HistoryRowData", (), {})()
    data._iter_count = row["iter"]
    data.solution = row["solution"]
    data.bound_diff = row["bound_diff"]
    data.comb_bound_diff = row["comb_bound_diff"]
    data.bound_half_width = row["bound_half_width"]
    data.bias_estimate = row["bias_estimate"]
    data.rmse_estimate = row["rmse_estimate"]
    data.rmse_tol = (
        row["rmse_tol"]
        if row["rmse_tol"] is not None
        else context.get("rmse_tol", context.get("target_rmse_tol"))
    )
    data.n_min = row["n_min"]
    data.n_total = row["n_total"]
    data.m = row["m"]
    data.abs_tol = context.get("abs_tol", None)
    data.rel_tol = context.get("rel_tol", None)
    if row["xfull.shape"] is not None:
        data.xfull = type("HistoryShape", (), {"shape": row["xfull.shape"]})()
    else:
        data.xfull = None
    data.stopping_crit = _build_history_stopping_criterion(history, stopping_criterion)
    return row["stage"], data


def print_iteration_log(
    history,
    stopping_criterion=None,
    printed_only=True,
    include_header=True,
    file=None,
):
    """Print a stored iteration log from an IterationHistoryTable."""
    if history is None or len(history) == 0:
        return
    if file is None:
        file = sys.stdout
    context = dict(getattr(history, "context", None) or {})
    label = context.get("trace_label", "")
    if (not label) and stopping_criterion is not None:
        label = str(getattr(stopping_criterion, "trace_label", ""))
    indices = range(len(history))
    if printed_only:
        indices = [index for index in indices if history["printed"][index]]
    indices = list(indices)
    if not indices:
        return
    with redirect_stdout(file):
        if include_header and label:
            print(f"=== {label} iteration log ===")
        for offset, index in enumerate(indices):
            stage, data = _build_history_data(history, index, stopping_criterion)
            print_diagnostic(
                stage,
                data,
                table_header=offset == 0,
                verbose=True,
                visible_columns=history.visible_columns,
            )


def format_iteration_log(
    history, stopping_criterion=None, printed_only=True, include_header=True
):
    """Return a stored iteration log as text."""
    stream = io.StringIO()
    print_iteration_log(
        history,
        stopping_criterion=stopping_criterion,
        printed_only=printed_only,
        include_header=include_header,
        file=stream,
    )
    return stream.getvalue().strip()


class _IterationTraceLogger(object):
    def __init__(self, stopping_criterion):
        """Create a trace logger bound to the given stopping criterion.

        Args:
            stopping_criterion: Stopping criterion instance. The logger reads
                the optional attributes ``trace_iterations`` (bool),
                ``trace_label`` (str), and ``verbose`` (bool)
                to configure itself.
        """
        self.enabled = bool(getattr(stopping_criterion, "trace_iterations", False))
        self.label = str(getattr(stopping_criterion, "trace_label", ""))
        self.verbose = bool(getattr(stopping_criterion, "verbose", True))
        self.print_enabled = bool(getattr(stopping_criterion, "trace_print", True))
        self.stopping_criterion = stopping_criterion
        self.header_printed = False
        self.table_header_printed = False
        self.iter_count = 0
        self.visible_columns = None
        self.pending_resume_signature = None
        self._last_iter_snapshot = None
        self._last_data = None
        self._last_iter_count = 0
        self._last_iter_history_index = None
        self._last_printed_iter_count = 0
        self.history = IterationHistoryTable() if self.enabled else None
        self.stopping_criterion.iteration_history = self.history
        if self.history is not None:
            self.history.context = _history_context_from_stopping_criterion(
                stopping_criterion
            )

    def _would_be_throttled(self, iter_count):
        """Return True if an ITER row with this count would be suppressed by throttling."""
        if self.verbose:
            return False
        if iter_count is None or iter_count <= _THROTTLE_ITER_THRESHOLD:
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
        if self.enabled and self.print_enabled and (not self.header_printed) and self.label:
            print(f"=== {self.label} iteration log ===")
            self.header_printed = True

    def _get_visible_columns(self, data, row=None):
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
        row = _extract_diagnostic_row(data) if row is None else row
        self.visible_columns = _visible_columns_from_row(row)
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
        row = _extract_diagnostic_row(data)
        visible_columns = self._get_visible_columns(data, row=row)
        printed = stage != "ITER" or not self._would_be_throttled(row["iter"])
        history_index = None
        if self.history is not None:
            history_index = self.history.append(
                stage,
                row,
                visible_columns=visible_columns,
                printed=printed,
            )
        self._last_data = data
        if stage == "ITER":
            self._last_iter_snapshot = data
            self._last_iter_count = row["iter"] if row["iter"] is not None else 0
            self._last_iter_history_index = history_index
            if printed:
                self._last_printed_iter_count = self._last_iter_count
        if self.print_enabled:
            print_diagnostic(
                stage,
                data,
                table_header=not self.table_header_printed,
                verbose=self.verbose,
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
        if self.print_enabled:
            print_diagnostic(
                "ITER",
                self._last_iter_snapshot,
                table_header=False,
                verbose=True,
                visible_columns=self.visible_columns,
            )
        if self.history is not None and self._last_iter_history_index is not None:
            self.history.mark_printed(self._last_iter_history_index, True)
        self._last_printed_iter_count = self._last_iter_count

    def finalize(self):
        """Force-print the last ITER row if throttling suppressed it, then clear snapshot."""
        self._flush_last_if_suppressed()
        if self.enabled and self._last_data is not None:
            self._last_data.iteration_history = self.history
        self._last_iter_snapshot = None


def print_diagnostic(
    label,
    data,
    table_header=False,
    verbose=True,
    visible_columns=None,
):
    """Print diagnostic information for an integration state.

    Args:
        label (str): Stage label shown in the first column.
        data (object): Integration state carrying fields such as ``solution``,
            ``n_total``, ``n_min``, ``m``, and ``xfull``.
        table_header (bool, optional): Whether to print the compact table
            header before the row. Defaults to False.
        verbose (bool, optional): Whether to print every ``ITER`` row.
            Defaults to True. If False, the current iteration-log throttling
            rules are applied.
        visible_columns (tuple[str, ...] | list[str] | None, optional): Ordered
            columns to print. Defaults to all supported columns.
    """
    row = _extract_diagnostic_row(data)
    solution_display = row["solution"]
    bound_diff_display = row["bound_diff"]
    comb_bound_diff_display = row["comb_bound_diff"]
    bound_half_width_display = row["bound_half_width"]
    bias_estimate_display = row["bias_estimate"]
    rmse_estimate_display = row["rmse_estimate"]
    rmse_tol_display = row["rmse_tol"]
    iter_display = row["iter"]
    n_min_display = row["n_min"]
    n_total_display = row["n_total"]
    m_display = row["m"]
    xfull_shape = row["xfull.shape"]

    sc = getattr(data, "stopping_crit", None)
    tolerance_candidates = []
    for tol_value in (
        getattr(data, "abs_tol", None),
        rmse_tol_display,
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
    n_min_formatted = (
        f"{n_min_display:>10}" if n_min_display is not None else f"{'None':>10}"
    )
    bound_diff_formatted = _format_bound(bound_diff_display)
    comb_bound_diff_formatted = _format_bound(comb_bound_diff_display)
    bound_half_width_formatted = _format_bound(bound_half_width_display)
    bias_estimate_formatted = _format_bound(bias_estimate_display)
    rmse_estimate_formatted = _format_bound(rmse_estimate_display)
    rmse_tol_formatted = _format_bound(rmse_tol_display)
    iter_formatted = f"{iter_display:>4}" if iter_display is not None else " " * 4
    m_formatted = f"{m_display:>6}" if m_display is not None else f"{'None':>6}"

    throttle = iter_display
    if (not verbose) and label == "ITER" and throttle is not None and throttle > _THROTTLE_ITER_THRESHOLD:
        step = 10 ** int(log10(throttle))
        if throttle % step != 0:
            return
    if visible_columns is None:
        visible_columns = _visible_columns_from_row(row, include_n_min=True)
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
