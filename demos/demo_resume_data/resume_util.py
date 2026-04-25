"""Utilities for resume demos and text-report generation."""

from contextlib import redirect_stdout
import io
from time import strftime

from qmcpy import CubQMCLatticeG, Genz, Lattice
from qmcpy.stopping_criterion import abstract_stopping_criterion as _asc


TRACE_ITERATIONS = True


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


def make_oscillatory_solver(
    abs_tol,
    rel_tol=0,
    seed=7,
    dimension=3,
    trace_label="",
    trace_iterations=True,
):
    """Build a CubQMCLatticeG solver for the standard oscillatory demo case."""
    integrand = Genz(
        Lattice(dimension=dimension, seed=seed),
        kind_func="oscillatory",
        kind_coeff=1,
    )
    solver = CubQMCLatticeG(integrand, abs_tol=abs_tol, rel_tol=rel_tol)
    solver.trace_iterations = trace_iterations
    solver.trace_label = trace_label
    return solver


def make_solver(
    abs_tol,
    rel_tol=0,
    seed=7,
    dimension=3,
    trace_label="",
    trace_iterations=None,
):
    """Build the standard oscillatory demo solver with shared defaults."""
    if trace_iterations is None:
        trace_iterations = TRACE_ITERATIONS
    return make_oscillatory_solver(
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        seed=seed,
        dimension=dimension,
        trace_label=trace_label,
        trace_iterations=trace_iterations,
    )


def half_width(data):
    """Return the interval half-width from a QMCPy data object."""
    return (data.comb_bound_high.item() - data.comb_bound_low.item()) / 2


def print_stage_summary(rows, title="Stage summary"):
    """Print compact stage-by-stage sample/time summary rows."""
    sep = "-" * 55
    print(f"\n{title}")
    print(sep)
    print(
        f"{'Stage':<7} {'abs_tol':>7} {'total n':>9} {'new n':>9} {'half-width':>10} {'time (s)':>8}"
    )
    print(sep)
    for name, abs_tol, total_n, new_n, hw, tsec in rows:
        print(
            f"{name:<7} {abs_tol:>7.0e} {int(total_n):>9,} {int(new_n):>9,} {hw:>10.2e} {tsec:>8.4f}"
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
    """Print a compact block of resume-vs-fresh comparison metrics."""
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
    """Run ``integrate`` while capturing printed diagnostic output.

    Args:
        stopping_criterion: Stopping criterion instance.
        *args: Positional arguments forwarded to ``integrate``.
        **kwargs: Keyword arguments forwarded to ``integrate``.

    Returns:
        tuple[object, str]: ``integrate`` return value and captured stdout.
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


def _run_logged_case(factory, label, throttle_iterations=True, **integrate_kwargs):
    sc = enable_diagnostics(
        factory(), label, throttle_iterations=throttle_iterations
    )
    (solution, data), log = capture_integrate(sc, **integrate_kwargs)
    return solution, data, log.strip()


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
        sol1, data1, loose_log = _run_logged_case(
            case["loose"], f"{name}-LOOSE", throttle_iterations=throttle_iterations
        )
        old_n = int(getattr(data1, "n_total", 0))
        sol2, data2, resume_log = _run_logged_case(
            case["tight"],
            f"{name}-RESUME",
            throttle_iterations=throttle_iterations,
            resume=data1,
        )
        new_n = int(getattr(data2, "n_total", 0)) - old_n
        row.update(
            {
                "status": "ok",
                "loose_solution": format_value(sol1),
                "resume_solution": format_value(sol2),
                "old_n_total": str(old_n),
                "resume_n_total": str(int(getattr(data2, "n_total", 0))),
                "resume_n_new": str(new_n),
                "resume_time": format_value(
                    getattr(data2, "time_integrate", float("nan")), ndigits=4
                ),
                "loose_log": loose_log,
                "resume_log": resume_log,
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
        sol, data, fresh_log = _run_logged_case(
            case["tight"], f"{name}-FRESH", throttle_iterations=throttle_iterations
        )
        row.update(
            {
                "status": "ok",
                "fresh_solution": format_value(sol),
                "fresh_n_total": str(int(getattr(data, "n_total", 0))),
                "fresh_time": format_value(
                    getattr(data, "time_integrate", float("nan")), ndigits=4
                ),
                "fresh_log": fresh_log,
            }
        )
    except Exception as exc:
        row.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
    return row


def write_report(path, title, rows, summary_keys=(), log_sections=()):
    """Write a text report for resume-demo rows.

    Args:
        path: Output path object.
        title (str): Report title.
        rows (list[dict]): Case result rows.
        summary_keys (tuple[str, ...] | list[str], optional): Summary fields to
            print before the logs. Defaults to an empty tuple.
        log_sections (tuple[tuple[str, str], ...] | list[tuple[str, str]], optional):
            Pairs of display label and row key for embedded logs. Defaults to an
            empty tuple.
    """
    lines = [title, f"generated: {strftime('%Y-%m-%d %H:%M:%S')}", ""]
    for row in rows:
        lines.append(
            f"[{row.get('name', 'unknown')}] status={row.get('status', 'unknown')}"
        )
        for key in summary_keys:
            if key in row:
                lines.append(f"  {key}: {row[key]}")
        if summary_keys:
            lines.append("")
        printed_log = False
        for section_label, row_key in log_sections:
            log_text = row.get(row_key, "")
            if not log_text:
                continue
            if printed_log:
                lines.append("")
            lines.append(f"  {section_label}:")
            lines.extend([f"    {line}" for line in log_text.splitlines()])
            printed_log = True
        if "error" in row:
            if printed_log or summary_keys:
                lines.append("")
            lines.append(f"  error: {row['error']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
