"""Utilities for resume demos and text-report generation."""

from contextlib import redirect_stdout
import io
import math
from time import strftime

from qmcpy import CubQMCLatticeG, Genz, Lattice
from qmcpy.stopping_criterion import abstract_stopping_criterion as _asc


TRACE_ITERATIONS = True
TRACE_THROTTLE_ITERATIONS = True


def format_value(value, ndigits=8):
    """Format a scalar-like value for report output.

    Args:
        value: Value to format.
        ndigits (int, optional): Digits after the decimal point. Defaults to 8.

    Returns:
        str: Formatted value string.
    """
    try:
        return f"{float(value):.{ndigits}f}"
    except Exception:
        return str(value)


def _format_problem_inputs(stopping_criterion):
    """Build a compact problem-input summary for a stopping criterion.

    Args:
        stopping_criterion: Stopping criterion instance.

    Returns:
        str: Multi-line text block describing key solver inputs.
    """
    integrand = getattr(stopping_criterion, "integrand", None)
    lines = []

    def add_scalar(label, value, *, ndigits=6):
        if value is None:
            return
        try:
            fval = float(value)
            if fval != 0 and abs(fval) < 10 ** (-ndigits):
                formatted = f"{fval:.2e}"
            else:
                formatted = format_value(value, ndigits=ndigits)
        except (TypeError, ValueError):
            formatted = str(value)
        lines.append(f"{label}: {formatted}")

    def add_int(label, value):
        if value is None:
            return
        try:
            lines.append(f"{label}: {int(value)}")
        except Exception:
            lines.append(f"{label}: {value}")

    add_scalar("abs_tol", getattr(stopping_criterion, "abs_tol", None))
    add_scalar("rel_tol", getattr(stopping_criterion, "rel_tol", None))
    _target_rmse_tol = getattr(stopping_criterion, "target_rmse_tol", None)
    if _target_rmse_tol is not None:
        add_scalar("target_rmse_tol", _target_rmse_tol)
    else:
        add_scalar("rmse_tol", getattr(stopping_criterion, "rmse_tol", None))
    add_int("n_init", getattr(stopping_criterion, "n_init", None))
    add_int("n_limit", getattr(stopping_criterion, "n_limit", None))
    add_int("levels_min", getattr(stopping_criterion, "levels_min", None))
    add_int("levels_max", getattr(stopping_criterion, "levels_max", None))
    if integrand is not None:
        add_int("dimension", getattr(integrand, "d", None))
    return "\n".join(lines)


def make_CubQMCLatticeG_solver(
    abs_tol,
    rel_tol=0,
    seed=7,
    dimension=3,
    trace_label="",
    trace_iterations=None,
    trace_throttle_iterations=None,
):
    """Build a CubQMCLatticeG solver for the demo case.

    Args:
        abs_tol (float): Absolute error tolerance.
        rel_tol (float, optional): Relative error tolerance. Defaults to 0.
        seed (int, optional): Random seed for the lattice. Defaults to 7.
        dimension (int, optional): Integrand dimension. Defaults to 3.
        trace_label (str, optional): Label for the iteration log. Defaults to
            an empty string (no header printed).
        trace_iterations (bool | None, optional): Whether to enable iteration
            logging. If None, falls back to :data:`TRACE_ITERATIONS`.
        trace_throttle_iterations (bool | None, optional): Whether to throttle
            printed iteration rows. If None, falls back to
            :data:`TRACE_THROTTLE_ITERATIONS`.

    Returns:
        CubQMCLatticeG: Configured stopping criterion instance.
    """
    integrand = Genz(
        Lattice(dimension=dimension, seed=seed),
        kind_func="oscillatory",
        kind_coeff=1,
    )
    if trace_iterations is None:
        trace_iterations = TRACE_ITERATIONS
    if trace_throttle_iterations is None:
        trace_throttle_iterations = TRACE_THROTTLE_ITERATIONS
    solver = CubQMCLatticeG(integrand, abs_tol=abs_tol, rel_tol=rel_tol)
    solver.trace_iterations = trace_iterations
    solver.trace_label = trace_label
    solver.trace_throttle_iterations = trace_throttle_iterations
    return solver


def half_width(data):
    """Return the confidence-interval half-width from a QMCPy data object.

    Computed as ``(comb_bound_high - comb_bound_low) / 2``, which equals
    ``z* * inflate * sigma_hat / sqrt(n)`` for CLT-based criteria.

    Args:
        data (Data): Completed integration data object with ``comb_bound_high``
            and ``comb_bound_low`` attributes.

    Returns:
        float: Half-width of the confidence interval.
    """
    return (data.comb_bound_high.item() - data.comb_bound_low.item()) / 2


def print_integration_result(
    stage_name,
    solution,
    data,
    *,
    previous_n_total=None,
    time_value=None,
    time_label="Time",
    additional_time_pairs=(),
):
    """Print a consistent integration-result block for resume demos.

    Args:
        stage_name (str): Human-readable stage label, e.g. ``"Resumed"``.
        solution: Scalar-like solution returned by ``integrate``.
        data: QMCPy ``Data`` object with ``n_total`` and time/bound attributes.
        previous_n_total (int | None, optional): Previous sample count used for
            resume runs. When provided, previous/total/new sample lines are
            printed. Defaults to None.
        time_value (float | None, optional): Time value to print for the primary
            time line. Defaults to ``data.time_integrate``.
        time_label (str, optional): Label for the primary time line.
            Defaults to ``"Time"``.
        additional_time_pairs (iterable[tuple[str, float]], optional): Extra
            ``(label, seconds)`` time lines to print after the primary one.
            Defaults to an empty tuple.
    """
    total_n = int(getattr(data, "n_total", 0))

    def _positive_float(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value) or value <= 0:
            return None
        return value

    sc = getattr(data, "stopping_crit", None)
    tolerance_candidates = []
    for tol_value in (
        getattr(data, "abs_tol", None),
        getattr(data, "rmse_tol", None),
        getattr(sc, "abs_tol", None) if sc is not None else None,
        getattr(sc, "rmse_tol", None) if sc is not None else None,
    ):
        tol_float = _positive_float(tol_value)
        if tol_float is not None:
            tolerance_candidates.append(tol_float)

    solution_decimals = 8
    if tolerance_candidates:
        solution_decimals = int(math.ceil(-math.log10(min(tolerance_candidates)))) + 1
        solution_decimals = max(7, min(12, solution_decimals))

    value_w = max(12, solution_decimals + 3)
    primary_time = (
        float(getattr(data, "time_integrate", float("nan")))
        if time_value is None
        else float(time_value)
    )

    labels = [
        f"{stage_name} solution",
        "Samples used" if previous_n_total is None else "Previous samples",
        "Total samples" if previous_n_total is not None else None,
        "New samples" if previous_n_total is not None else None,
        time_label,
        *[str(extra_label) for extra_label, _ in additional_time_pairs],
        "Estimated interval half-width",
    ]
    labels = [label for label in labels if label is not None]
    label_w = max(len(label) + 1 for label in labels)

    def _print_value_line(label, value_text):
        print(f"  {label + ':':<{label_w}} {value_text:>{value_w}}")

    print()
    _print_value_line(
        f"{stage_name} solution", f"{solution.item():.{solution_decimals}f}"
    )
    if previous_n_total is None:
        _print_value_line("Samples used", f"{total_n:,}")
    else:
        previous_n_total = int(previous_n_total)
        _print_value_line("Previous samples", f"{previous_n_total:,}")
        _print_value_line("Total samples", f"{total_n:,}")
        _print_value_line("New samples", f"{total_n - previous_n_total:,}")

    _print_value_line(time_label, f"{primary_time:.4f} s")
    for extra_label, extra_time in additional_time_pairs:
        _print_value_line(str(extra_label), f"{float(extra_time):.4f} s")
    _print_value_line("Estimated interval half-width", f"{half_width(data):.2e}")


def print_stage_summary(rows, title="Stage summary", tol_header="abs_tol"):
    """Print a compact stage-by-stage sample and time summary table.

    Args:
        rows (list[tuple]): Rows of the form
            ``(name, abs_tol, total_n, new_n, iters, solution, half_width, time_sec)``.
            ``iters`` may be ``None`` when iteration tracing is disabled.
        title (str, optional): Table title. Defaults to ``"Stage summary"``.
        tol_header (str, optional): Column header for the tolerance column.
            Defaults to ``"abs_tol"``.
    """
    tol_values = [
        float(abs_tol)
        for _, abs_tol, *_ in rows
        if abs_tol is not None and float(abs_tol) > 0
    ]
    if tol_values:
        solution_decimals = int(math.ceil(-math.log10(min(tol_values)))) + 1
        solution_decimals = max(7, min(12, solution_decimals))
    else:
        solution_decimals = 8
    sol_w = solution_decimals + 4  # sign + leading digit + dot + decimals
    stage_w = 7
    # Keep rmse_tol values visually separated by one extra leading space.
    tol_w = max(7, len(tol_header) + (1 if tol_header == "rmse_tol" else 0))
    total_n_w = 9
    new_n_w = 9
    iters_w = 6
    hw_w = 10
    time_w = 8
    sep_len = stage_w + tol_w + total_n_w + new_n_w + iters_w + sol_w + hw_w + time_w + 7
    sep = "-" * sep_len
    print(f"\n{title}")
    print(sep)
    print(f"{'Stage':<{stage_w}} {tol_header:>{tol_w}} {'total n':>{total_n_w}} {'new n':>{new_n_w}}"
          f" {'iters':>{iters_w}} {'solution':>{sol_w}} {'half-width':>{hw_w}} {'time (s)':>{time_w}}")
    print(sep)
    for name, abs_tol, total_n, new_n, iters, solution, hw, tsec in rows:
        iter_str = str(int(iters)) if iters is not None else "-"
        sol_str = f"{float(solution):.{solution_decimals}f}"
        abs_tol_str = f"{float(abs_tol):.0e}" if abs_tol is not None else "---"
        print(f"{name:<{stage_w}} {abs_tol_str:>{tol_w}} {int(total_n):>{total_n_w},} {int(new_n):>{new_n_w},}"
              f" {iter_str:>{iters_w}} {sol_str:>{sol_w}} {hw:>{hw_w}.2e} {tsec:>{time_w}.4f}"
        )
    print(sep)


def print_comparison_metrics(
    incremental_speedup,
    new_samples_resume,
    new_samples_fresh,
    samples_saved,
    two_step_time,
    fresh_wall_time,
    label_w=36,
    val_w=14,
):
    """Print a compact block of resume-vs-fresh comparison metrics.

    Args:
        incremental_speedup (float): Ratio of fresh new-sample count to resume
            new-sample count (higher is better for resume).
        new_samples_resume (int): New samples drawn during the resume stage.
        new_samples_fresh (int): New samples drawn during a fresh tight run.
        samples_saved (int): Difference ``new_samples_fresh - new_samples_resume``.
        two_step_time (float): Wall time of the loose + resume two-stage run.
        fresh_wall_time (float): Wall time of the fresh tight run.
        label_w (int, optional): Width of the label column. Defaults to 36.
        val_w (int, optional): Width of the value column. Defaults to 14.
    """
    print("\nComparison metrics")
    print(
        f"{'Incremental speedup (fresh/resume):':<{label_w}} {incremental_speedup:>{val_w}.2f}x"
    )
    print(f"{'Resume new samples:':<{label_w}} {new_samples_resume:>{val_w},}")
    print(f"{'Fresh new samples:':<{label_w}} {new_samples_fresh:>{val_w},}")
    print(f"{'Samples saved by resume:':<{label_w}} {samples_saved:>{val_w},}")
    print(f"{'End-to-end loose+resume time:':<{label_w}} {two_step_time:>{val_w}.4f} s")
    print(f"{'Fresh tight time:':<{label_w}} {fresh_wall_time:>{val_w}.4f} s")


def enable_diagnostics(stopping_criterion, label, throttle_iterations=True):
    """Enable iteration diagnostics on a stopping criterion instance.

    Args:
        stopping_criterion: Stopping criterion instance to modify.
        label (str): Label shown in the iteration log.
        throttle_iterations (bool, optional): Whether to throttle printed
            iteration rows. Defaults to True.

    Returns:
        object: The same stopping criterion instance.
    """
    setattr(stopping_criterion, "trace_iterations", True)
    setattr(stopping_criterion, "trace_label", label)
    setattr(stopping_criterion, "trace_throttle_iterations", throttle_iterations)
    return stopping_criterion


def capture_integrate(stopping_criterion, *args, **kwargs):
    """Run ``integrate`` while capturing all printed diagnostic output.

    Args:
        stopping_criterion: Stopping criterion instance.
        *args: Positional arguments forwarded to ``integrate``.
        **kwargs: Keyword arguments forwarded to ``integrate``.

    Returns:
        tuple[tuple, str]: ``((solution, data), captured_stdout)`` where
        *captured_stdout* contains the full text printed during integration.
    """
    stream = io.StringIO()
    with redirect_stdout(stream):
        out = stopping_criterion.integrate(*args, **kwargs)
    return out, stream.getvalue()


def make_case(name, loose_factory, tight_factory):
    """Create a standard loose/tight demo case record.

    Args:
        name (str): Human-readable stopping criterion name.
        loose_factory (callable): Zero-argument factory for the loose run.
        tight_factory (callable): Zero-argument factory for the tight run.

    Returns:
        dict: Case metadata consumed by the demo runner.
    """
    return {"name": name, "loose": loose_factory, "tight": tight_factory}


def make_tol_case(name, builder, loose_tol, tight_tol):
    """Create a case from a scalar-tolerance builder.

    Args:
        name (str): Human-readable stopping criterion name.
        builder (callable): Single-argument factory taking a tolerance value.
        loose_tol (float): Tolerance for the loose run.
        tight_tol (float): Tolerance for the tight run.

    Returns:
        dict: Case metadata consumed by the demo runner.
    """
    return make_case(name, lambda: builder(loose_tol), lambda: builder(tight_tol))


def make_abs_tol_builder(
    sc_class,
    integrand_factory,
    *,
    rel_tol=0,
    n_init=None,
    n_limit=None,
):
    """Build a factory for stopping criteria driven by ``abs_tol``.

    Args:
        sc_class: Stopping criterion class.
        integrand_factory (callable): Zero-argument integrand factory.
        rel_tol (float, optional): Relative tolerance. Defaults to 0.
        n_init (int | None, optional): Initial sample count. Defaults to None.
        n_limit (int | None, optional): Sample budget. Defaults to None.

    Returns:
        callable: Builder taking ``abs_tol`` and returning a stopping criterion.
    """
    def build(abs_tol):
        kwargs = {"abs_tol": abs_tol, "rel_tol": rel_tol}
        if n_init is not None:
            kwargs["n_init"] = n_init
        if n_limit is not None:
            kwargs["n_limit"] = n_limit
        return sc_class(integrand_factory(), **kwargs)

    return build


def make_named_tol_builder(sc_class, integrand_factory, tol_name, **fixed_kwargs):
    """Build a factory for stopping criteria with a named tolerance argument.

    Args:
        sc_class: Stopping criterion class.
        integrand_factory (callable): Zero-argument integrand factory.
        tol_name (str): Tolerance keyword to set, such as ``abs_tol`` or
            ``rmse_tol``.
        **fixed_kwargs: Additional keyword arguments passed to the class.

    Returns:
        callable: Builder taking the tolerance value and returning a stopping
        criterion.
    """
    def build(tol):
        return sc_class(integrand_factory(), **{tol_name: tol, **fixed_kwargs})

    return build


def _safe_half_width(data):
    """Return half-width from data, or bias_estimate for MLQMC types,
    or RMSE estimate for MLMC types, or NaN when nothing is available."""
    try:
        return (data.comb_bound_high.item() - data.comb_bound_low.item()) / 2
    except Exception:
        pass
    try:
        return float(data.bias_estimate)
    except Exception:
        pass
    try:
        import numpy as np
        n = data.n_level
        v = data.var_level
        # Align lengths: var_level may be shorter than n_level by one (last level index)
        min_len = min(len(n), len(v))
        n, v = n[:min_len], v[:min_len]
        valid = n > 0
        return float(np.sqrt(np.sum(v[valid] / n[valid])))
    except Exception:
        return float("nan")


def _get_sc_tol(sc):
    """Return (value, name) for the primary tolerance of a stopping criterion.

    Prefers ``target_rmse_tol`` (Cont solvers), then ``abs_tol``, then ``rmse_tol``.
    Returns ``(None, None)`` when nothing is found.
    """
    for attr in ("target_rmse_tol", "abs_tol", "rmse_tol"):
        try:
            val = float(getattr(sc, attr, None))
            if val > 0:
                return val, attr
        except (TypeError, ValueError):
            pass
    return None, None


def _run_logged_case(factory, label, throttle_iterations=True, **integrate_kwargs):
    """Build, enable diagnostics on, and integrate a stopping criterion.

    Args:
        factory (callable): Zero-argument factory returning a stopping criterion.
        label (str): Iteration-log label forwarded to :func:`enable_diagnostics`.
        throttle_iterations (bool, optional): Whether to throttle printed rows.
            Defaults to True.
        **integrate_kwargs: Extra keyword arguments forwarded to ``integrate``.

    Returns:
        tuple: ``(solution, data, log_text, input_text, sc)`` where *log_text* is
        the captured stdout from the integration run, *input_text* is a
        compact summary of the stopping criterion inputs, and *sc* is the
        stopping criterion instance after integration.
    """
    sc = enable_diagnostics(
        factory(), label, throttle_iterations=throttle_iterations
    )
    (solution, data), log = capture_integrate(sc, **integrate_kwargs)
    return solution, data, log.strip(), _format_problem_inputs(sc), sc


def run_resume_case(case, throttle_iterations=True):
    """Run a loose solve followed by a resumed tighter solve.

    Args:
        case (dict): Case definition with ``name``, ``loose``, and ``tight``.
        throttle_iterations (bool, optional): Whether to throttle printed
            iteration rows. Defaults to True.

    Returns:
        dict: Result record used by the report writers.
    """
    name = case["name"]
    row = {"name": name}
    try:
        sol1, data1, loose_log, loose_inputs, sc1 = _run_logged_case(
            case["loose"], f"{name}-LOOSE", throttle_iterations=throttle_iterations
        )
        old_n = int(getattr(data1, "n_total", 0))
        # Capture loose stats before passing data1 into resume (may mutate in-place)
        loose_iters = getattr(data1, "_iter_count", None)
        loose_hw = _safe_half_width(data1)
        loose_time_f = float(getattr(data1, "time_integrate", 0.0))
        loose_solution_f = float(sol1.item())
        sol2, data2, resume_log, resume_inputs, sc2 = _run_logged_case(
            case["tight"],
            f"{name}-RESUME",
            throttle_iterations=throttle_iterations,
            resume=data1,
        )
        resume_n = int(getattr(data2, "n_total", 0))
        new_n = resume_n - old_n
        row.update(
            {
                "status": "ok",
                "loose_solution": format_value(sol1, ndigits=7),
                "resume_solution": format_value(sol2, ndigits=7),
                "old_n_total": str(old_n),
                "resume_n_total": str(resume_n),
                "resume_n_new": str(new_n),
                "resume_time": format_value(
                    getattr(data2, "time_integrate", float("nan")), ndigits=4
                ),
                "loose_inputs": loose_inputs,
                "resume_inputs": resume_inputs,
                "loose_log": loose_log,
                "resume_log": resume_log,
                # Extra numeric fields for stage summary
                "_loose_abs_tol": _get_sc_tol(sc1)[0],
                "_loose_tol_name": _get_sc_tol(sc1)[1],
                "_loose_n": old_n,
                "_loose_hw": loose_hw,
                "_loose_iters": loose_iters,
                "_loose_solution_f": loose_solution_f,
                "_loose_time_f": loose_time_f,
                "_resume_abs_tol": _get_sc_tol(sc2)[0],
                "_resume_tol_name": _get_sc_tol(sc2)[1],
                "_resume_n": resume_n,
                "_resume_hw": _safe_half_width(data2),
                "_resume_iters": getattr(data2, "_iter_count", None),
                "_resume_solution_f": float(sol2.item()),
                "_resume_time_f": float(getattr(data2, "time_integrate", 0.0)),
            }
        )
    except Exception as exc:
        row.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
    return row


def run_fresh_case(case, throttle_iterations=True):
    """Run a fresh tight solve from scratch.

    Args:
        case (dict): Case definition with ``name`` and ``tight``.
        throttle_iterations (bool, optional): Whether to throttle printed
            iteration rows. Defaults to True.

    Returns:
        dict: Result record used by the report writers.
    """
    name = case["name"]
    row = {"name": name}
    try:
        sol, data, fresh_log, fresh_inputs, sc = _run_logged_case(
            case["tight"], f"{name}-FRESH", throttle_iterations=throttle_iterations
        )
        fresh_n = int(getattr(data, "n_total", 0))
        row.update(
            {
                "status": "ok",
                "fresh_solution": format_value(sol, ndigits=7),
                "fresh_n_total": str(fresh_n),
                "fresh_time": format_value(
                    getattr(data, "time_integrate", float("nan")), ndigits=4
                ),
                "fresh_inputs": fresh_inputs,
                "fresh_log": fresh_log,
                # Extra numeric fields for stage summary
                "_fresh_abs_tol": _get_sc_tol(sc)[0],
                "_fresh_tol_name": _get_sc_tol(sc)[1],
                "_fresh_n": fresh_n,
                "_fresh_hw": _safe_half_width(data),
                "_fresh_iters": getattr(data, "_iter_count", None),
                "_fresh_solution_f": float(sol.item()),
                "_fresh_time_f": float(getattr(data, "time_integrate", 0.0)),
            }
        )
    except Exception as exc:
        row.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
    return row


def write_report(
    path,
    title,
    rows,
    summary_keys=(),
    input_sections=(),
    log_sections=(),
):
    """Write a text report for resume-demo rows.

    Args:
        path: Output path object.
        title (str): Report title.
        rows (list[dict]): Case result rows.
        summary_keys (tuple[str, ...] | list[str], optional): Summary fields to
            print after the logs. Defaults to an empty tuple.
        input_sections (tuple[tuple[str, str], ...] | list[tuple[str, str]], optional):
            Pairs of display label and row key for compact problem-input blocks.
            Defaults to an empty tuple.
        log_sections (tuple[tuple[str, str], ...] | list[tuple[str, str]], optional):
            Pairs of display label and row key for embedded logs. Defaults to an
            empty tuple.
    """
    separator = "~" * 60
    lines = [title, f"generated: {strftime('%Y-%m-%d %H:%M:%S')}", ""]
    for i, row in enumerate(rows):
        if i > 0:
            lines.append(separator)
            lines.append("")
        lines.append(
            f"[{row.get('name', 'unknown')}] status={row.get('status', 'unknown')}"
        )
        printed_inputs = False
        for section_label, row_key in input_sections:
            input_text = row.get(row_key, "")
            if not input_text:
                continue
            if printed_inputs:
                lines.append("")
            lines.append(f"  {section_label}:")
            lines.extend([f"    {line}" for line in input_text.splitlines()])
            printed_inputs = True
        printed_log = False
        for section_label, row_key in log_sections:
            log_text = row.get(row_key, "")
            if not log_text:
                continue
            if printed_inputs or printed_log:
                lines.append("")
            lines.append(f"  {section_label}:")
            lines.extend([f"    {line}" for line in log_text.splitlines()])
            printed_log = True
        if printed_log and summary_keys:
            lines.append("")
        printed_summary = False
        for key in summary_keys:
            if key in row:
                lines.append(f"  {key}: {row[key]}")
                printed_summary = True
        if "error" in row:
            if printed_log or printed_summary:
                lines.append("")
            lines.append(f"  error: {row['error']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_combined_report(path, title, resume_rows, fresh_rows):
    """Write a combined loose+resume and fresh report with per-example stage summaries.

    Each example block contains: loose inputs, resume inputs, fresh inputs,
    all three iteration logs, a stage summary table, and key numeric values.

    Args:
        path: Output path object.
        title (str): Report title.
        resume_rows (list[dict]): Rows returned by :func:`run_resume_case`.
        fresh_rows (list[dict]): Rows returned by :func:`run_fresh_case`,
            in the same order as *resume_rows*.
    """
    separator = "~" * 60
    lines = [title, f"generated: {strftime('%Y-%m-%d %H:%M:%S')}", ""]
    for i, (rrow, frow) in enumerate(zip(resume_rows, fresh_rows)):
        if i > 0:
            lines.append(separator)
            lines.append("")
        name = rrow.get("name", "unknown")
        rstatus = rrow.get("status", "unknown")
        fstatus = frow.get("status", "unknown")
        lines.append(f"[{name}] status={rstatus}")

        # Input sections
        for section_label, row_obj, row_key in [
            ("loose_inputs", rrow, "loose_inputs"),
            ("resume_inputs", rrow, "resume_inputs"),
            ("fresh_inputs", frow, "fresh_inputs"),
        ]:
            text = row_obj.get(row_key, "")
            if not text:
                continue
            lines.append("")
            lines.append(f"  {section_label}:")
            lines.extend([f"    {line}" for line in text.splitlines()])

        # Iteration log sections
        for section_label, row_obj, row_key in [
            ("loose_iteration_log", rrow, "loose_log"),
            ("resume_iteration_log", rrow, "resume_log"),
            ("fresh_iteration_log", frow, "fresh_log"),
        ]:
            text = row_obj.get(row_key, "")
            if not text:
                continue
            lines.append("")
            lines.append(f"  {section_label}:")
            lines.extend([f"    {line}" for line in text.splitlines()])

        # Stage summary table
        if rstatus == "ok" and fstatus == "ok":
            stage_rows = [
                (
                    "Loose",
                    rrow.get("_loose_abs_tol"),
                    rrow.get("_loose_n", 0),
                    rrow.get("_loose_n", 0),
                    rrow.get("_loose_iters"),
                    rrow.get("_loose_solution_f", float("nan")),
                    rrow.get("_loose_hw", float("nan")),
                    rrow.get("_loose_time_f", 0.0),
                ),
                (
                    "Resumed",
                    rrow.get("_resume_abs_tol"),
                    rrow.get("_resume_n", 0),
                    rrow.get("_resume_n", 0) - rrow.get("_loose_n", 0),
                    rrow.get("_resume_iters"),
                    rrow.get("_resume_solution_f", float("nan")),
                    rrow.get("_resume_hw", float("nan")),
                    rrow.get("_resume_time_f", 0.0),
                ),
                (
                    "Fresh",
                    frow.get("_fresh_abs_tol"),
                    frow.get("_fresh_n", 0),
                    frow.get("_fresh_n", 0),
                    frow.get("_fresh_iters"),
                    frow.get("_fresh_solution_f", float("nan")),
                    frow.get("_fresh_hw", float("nan")),
                    frow.get("_fresh_time_f", 0.0),
                ),
            ]
            # Use target_rmse_tol as the column header when all rows report that name
            tol_names = {
                rrow.get("_loose_tol_name"),
                rrow.get("_resume_tol_name"),
                frow.get("_fresh_tol_name"),
            } - {None}
            tol_header = "rmse_tol" if tol_names == {"target_rmse_tol"} else "abs_tol"
            stream = io.StringIO()
            with redirect_stdout(stream):
                print_stage_summary(stage_rows, title=f"Stage summary of {name}", tol_header=tol_header)
            summary_text = stream.getvalue().strip()
            lines.append("")
            lines.extend([f"  {line}" for line in summary_text.splitlines()])

        # Errors
        for row_obj in (rrow, frow):
            if "error" in row_obj:
                lines.append(f"  error: {row_obj['error']}")

        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
