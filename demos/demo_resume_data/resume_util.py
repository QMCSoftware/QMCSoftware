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
        lines.append(f"{label}: {format_value(value, ndigits=ndigits)}")

    def add_int(label, value):
        if value is None:
            return
        try:
            lines.append(f"{label}: {int(value)}")
        except Exception:
            lines.append(f"{label}: {value}")

    add_scalar("abs_tol", getattr(stopping_criterion, "abs_tol", None))
    add_scalar("rel_tol", getattr(stopping_criterion, "rel_tol", None))
    add_scalar("rmse_tol", getattr(stopping_criterion, "rmse_tol", None))
    add_int("n_init", getattr(stopping_criterion, "n_init", None))
    add_int("n_limit", getattr(stopping_criterion, "n_limit", None))
    add_int("levels_min", getattr(stopping_criterion, "levels_min", None))
    add_int("levels_max", getattr(stopping_criterion, "levels_max", None))
    if integrand is not None:
        add_int("dimension", getattr(integrand, "d", None))
    return "\n".join(lines)


def make_oscillatory_solver(
    abs_tol,
    rel_tol=0,
    seed=7,
    dimension=3,
    trace_label="",
    trace_iterations=True,
):
    """Build a CubQMCLatticeG solver for the standard oscillatory demo case.

    Args:
        abs_tol (float): Absolute error tolerance.
        rel_tol (float, optional): Relative error tolerance. Defaults to 0.
        seed (int, optional): Random seed for the lattice. Defaults to 7.
        dimension (int, optional): Integrand dimension. Defaults to 3.
        trace_label (str, optional): Label for the iteration log. Defaults to
            an empty string (no header printed).
        trace_iterations (bool, optional): Whether to enable iteration logging.
            Defaults to True.

    Returns:
        CubQMCLatticeG: Configured stopping criterion instance.
    """
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
    """Build the standard oscillatory demo solver with shared defaults.

    Thin wrapper around :func:`make_oscillatory_solver` that defaults
    ``trace_iterations`` to the module-level :data:`TRACE_ITERATIONS` flag.

    Args:
        abs_tol (float): Absolute error tolerance.
        rel_tol (float, optional): Relative error tolerance. Defaults to 0.
        seed (int, optional): Random seed for the lattice. Defaults to 7.
        dimension (int, optional): Integrand dimension. Defaults to 3.
        trace_label (str, optional): Label for the iteration log. Defaults to
            an empty string.
        trace_iterations (bool | None, optional): Whether to enable iteration
            logging.  If None, falls back to :data:`TRACE_ITERATIONS`.

    Returns:
        CubQMCLatticeG: Configured stopping criterion instance.
    """
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


def print_stage_summary(rows, title="Stage summary"):
    """Print a compact stage-by-stage sample and time summary table.

    Args:
        rows (list[tuple]): Rows of the form
            ``(name, abs_tol, total_n, new_n, half_width, time_sec)``.
        title (str, optional): Table title. Defaults to ``"Stage summary"``.
    """
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


def _run_logged_case(factory, label, throttle_iterations=True, **integrate_kwargs):
    """Build, enable diagnostics on, and integrate a stopping criterion.

    Args:
        factory (callable): Zero-argument factory returning a stopping criterion.
        label (str): Iteration-log label forwarded to :func:`enable_diagnostics`.
        throttle_iterations (bool, optional): Whether to throttle printed rows.
            Defaults to True.
        **integrate_kwargs: Extra keyword arguments forwarded to ``integrate``.

    Returns:
        tuple: ``(solution, data, log_text, input_text)`` where *log_text* is
        the captured stdout from the integration run and *input_text* is a
        compact summary of the stopping criterion inputs.
    """
    sc = enable_diagnostics(
        factory(), label, throttle_iterations=throttle_iterations
    )
    (solution, data), log = capture_integrate(sc, **integrate_kwargs)
    return solution, data, log.strip(), _format_problem_inputs(sc)


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
        sol1, data1, loose_log, loose_inputs = _run_logged_case(
            case["loose"], f"{name}-LOOSE", throttle_iterations=throttle_iterations
        )
        old_n = int(getattr(data1, "n_total", 0))
        sol2, data2, resume_log, resume_inputs = _run_logged_case(
            case["tight"],
            f"{name}-RESUME",
            throttle_iterations=throttle_iterations,
            resume=data1,
        )
        new_n = int(getattr(data2, "n_total", 0)) - old_n
        row.update(
            {
                "status": "ok",
                "loose_solution": format_value(sol1, ndigits=7),
                "resume_solution": format_value(sol2, ndigits=7),
                "old_n_total": str(old_n),
                "resume_n_total": str(int(getattr(data2, "n_total", 0))),
                "resume_n_new": str(new_n),
                "resume_time": format_value(
                    getattr(data2, "time_integrate", float("nan")), ndigits=4
                ),
                "loose_inputs": loose_inputs,
                "resume_inputs": resume_inputs,
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
        sol, data, fresh_log, fresh_inputs = _run_logged_case(
            case["tight"], f"{name}-FRESH", throttle_iterations=throttle_iterations
        )
        row.update(
            {
                "status": "ok",
                "fresh_solution": format_value(sol, ndigits=7),
                "fresh_n_total": str(int(getattr(data, "n_total", 0))),
                "fresh_time": format_value(
                    getattr(data, "time_integrate", float("nan")), ndigits=4
                ),
                "fresh_inputs": fresh_inputs,
                "fresh_log": fresh_log,
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
    lines = [title, f"generated: {strftime('%Y-%m-%d %H:%M:%S')}", ""]
    for row in rows:
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
