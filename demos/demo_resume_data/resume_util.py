"""Utilities for resume demos and text-report generation."""

import math
from time import strftime

#################################################################
# Formatting and reporting helpers
#################################################################
def format_value(value, ndigits=8):
    """Format a scalar-like value for report output."""
    try:
        return f"{float(value):.{ndigits}f}"
    except Exception:
        return str(value)


def _format_problem_inputs(stopping_criterion):
    """Build a compact problem-input summary for a stopping criterion."""
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

#################################################################
# Result summaries and diagnostics
#################################################################
def half_width(data):
    """Return the confidence-interval half-width from a QMCPy data object."""
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
    """Print a consistent integration-result block for resume demos."""
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
    """Print a compact stage-by-stage sample and time summary table."""
    print(format_stage_summary(rows, title=title, tol_header=tol_header))


def format_stage_summary(rows, title="Stage summary", tol_header="abs_tol"):
    """Return a compact stage-by-stage sample and time summary table."""
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
    lines = [
        f"\n{title}",
        sep,
        f"{'Stage':<{stage_w}} {tol_header:>{tol_w}} {'total n':>{total_n_w}} {'new n':>{new_n_w}}"
        f" {'iters':>{iters_w}} {'solution':>{sol_w}} {'half-width':>{hw_w}} {'time (s)':>{time_w}}",
        sep,
    ]
    for name, abs_tol, total_n, new_n, iters, solution, hw, tsec in rows:
        iter_str = str(int(iters)) if iters is not None else "-"
        sol_str = f"{float(solution):.{solution_decimals}f}"
        abs_tol_str = f"{float(abs_tol):.0e}" if abs_tol is not None else "---"
        lines.append(
            f"{name:<{stage_w}} {abs_tol_str:>{tol_w}} {int(total_n):>{total_n_w},} {int(new_n):>{new_n_w},}"
            f" {iter_str:>{iters_w}} {sol_str:>{sol_w}} {hw:>{hw_w}.2e} {tsec:>{time_w}.4f}"
        )
    lines.append(sep)
    return "\n".join(lines)


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
    print(f"{'Incremental speedup (fresh/resume):':<{label_w}} {incremental_speedup:>{val_w}.2f}x")
    print(f"{'Resume new samples:':<{label_w}} {new_samples_resume:>{val_w},}")
    print(f"{'Fresh new samples:':<{label_w}} {new_samples_fresh:>{val_w},}")
    print(f"{'Samples saved by resume:':<{label_w}} {samples_saved:>{val_w},}")
    print(f"{'End-to-end loose+resume time:':<{label_w}} {two_step_time:>{val_w}.4f} s")
    print(f"{'Fresh tight time:':<{label_w}} {fresh_wall_time:>{val_w}.4f} s")


def collect_resume_fresh_warnings(name, resume_stage, fresh_stage):
    """Return warning strings for mismatched resume-vs-fresh outcomes."""
    warning_lines = []

    resume_n = int(resume_stage.get("total_n", 0))
    fresh_n = int(fresh_stage.get("total_n", 0))
    if resume_n != fresh_n:
        warning_lines.append(
            f"WARNING: {name}: Inconsistent total samples across stages "
            f"(resume_n={resume_n}, fresh_n={fresh_n})"
        )

    try:
        resume_sol = float(resume_stage.get("solution", float("nan")))
        fresh_sol = float(fresh_stage.get("solution", float("nan")))
        resume_tol = float(resume_stage.get("tol", float("nan")))
    except (TypeError, ValueError):
        return warning_lines
    if math.isfinite(resume_sol) and math.isfinite(fresh_sol) and math.isfinite(resume_tol):
        if abs(resume_sol - fresh_sol) > 2 * resume_tol:
            warning_lines.append(
                f"WARNING: {name}: Resume and fresh solutions differ by more than 2 * tol "
                f"(resume_sol={resume_sol}, fresh_sol={fresh_sol}, tol={resume_tol})"
            )
    return warning_lines


def enable_diagnostics(stopping_criterion, label, verbose=False, print_live=False):
    """Enable iteration diagnostics on a stopping criterion instance."""
    setattr(stopping_criterion, "trace_iterations", True)
    setattr(stopping_criterion, "trace_label", label)
    setattr(stopping_criterion, "verbose", verbose)
    setattr(stopping_criterion, "trace_print", print_live)
    return stopping_criterion

#################################################################
#  Case factories and runners
#################################################################
def make_case(name, loose_factory, tight_factory):
    """Create a standard loose/tight demo case record."""
    return {"name": name, "loose": loose_factory, "tight": tight_factory}


def make_tol_case(name, builder, loose_tol, tight_tol):
    """Create a case from a scalar-tolerance builder."""
    return make_case(name, lambda: builder(loose_tol), lambda: builder(tight_tol))


def make_abs_tol_builder(sc_class, integrand_factory, *, rel_tol=0, n_init=None, n_limit=None):
    """Build a factory for stopping criteria driven by ``abs_tol``."""
    def build(abs_tol):
        kwargs = {"abs_tol": abs_tol, "rel_tol": rel_tol}
        if n_init is not None:
            kwargs["n_init"] = n_init
        if n_limit is not None:
            kwargs["n_limit"] = n_limit
        return sc_class(integrand_factory(), **kwargs)

    return build


def make_named_tol_builder(sc_class, integrand_factory, tol_name, **fixed_kwargs):
    """Build a factory for stopping criteria with a named tolerance argument."""
    def build(tol):
        return sc_class(integrand_factory(), **{tol_name: tol, **fixed_kwargs})

    return build


def _safe_half_width(data):
    """Return the best available half-width or RMSE estimate; otherwise NaN."""
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
    """Return the primary positive tolerance value and its attribute name."""
    for attr in ("target_rmse_tol", "abs_tol", "rmse_tol"):
        try:
            val = float(getattr(sc, attr, None))
            if val > 0:
                return val, attr
        except (TypeError, ValueError):
            pass
    return None, None


def _solution_value(solution):
    """Return a scalar float from a QMCPy solution object."""
    try:
        return float(solution.item())
    except Exception:
        return float(solution)


def _make_stage_record(name, sc, solution, data, inputs, previous_n_total=0):
    """Build a compact stage record from a completed integration."""
    total_n = int(getattr(data, "n_total", 0))
    tol, tol_name = _get_sc_tol(sc)
    return {
        "name": name,
        "tol": tol,
        "tol_name": tol_name,
        "total_n": total_n,
        "new_n": total_n - int(previous_n_total),
        "iters": getattr(data, "_iter_count", None),
        "solution": _solution_value(solution),
        "half_width": _safe_half_width(data),
        "time": float(getattr(data, "time_integrate", 0.0)),
        "inputs": inputs,
        "iteration_log": sc.format_iteration_log(
            history=getattr(data, "iteration_history", None)
        ),
    }


def _run_logged_case(factory, label, verbose=False, **integrate_kwargs):
    """Build, trace, integrate, and return the run outputs plus compact inputs."""
    sc = enable_diagnostics(factory(), label, verbose=verbose)
    solution, data = sc.integrate(**integrate_kwargs)
    return solution, data, _format_problem_inputs(sc), sc


def run_resume_case(case, verbose=False):
    """Run a loose solve followed by a resumed tighter solve."""
    name = case["name"]
    row = {"name": name}
    try:
        sol1, data1, loose_inputs, sc1 = _run_logged_case(
            case["loose"], f"{name}-LOOSE", verbose=verbose)
        loose_stage = _make_stage_record("Loose", sc1, sol1, data1, loose_inputs)
        old_n = loose_stage["total_n"]

        # Start RESUMED stage with the same solver, retuned to the tight tolerance.
        tight_sc = case["tight"]()
        if hasattr(tight_sc, "target_rmse_tol"):
            sc1.target_rmse_tol = float(tight_sc.target_rmse_tol)
        sc1.set_tolerance(
            abs_tol=getattr(tight_sc, "abs_tol", None),
            rel_tol=getattr(tight_sc, "rel_tol", None),
            rmse_tol=getattr(tight_sc, "target_rmse_tol", 
                             getattr(tight_sc, "rmse_tol", None) if not hasattr(tight_sc, "abs_tol") else None,),
        )
        setattr(sc1, "trace_label", f"{name}-RESUME")
        setattr(sc1, "verbose", verbose)
        sol2, data2 = sc1.integrate(resume=data1)
        resume_inputs = _format_problem_inputs(sc1)
        resume_stage = _make_stage_record(
            "Resumed", sc1, sol2, data2, resume_inputs, previous_n_total=old_n
        )
        row.update(
            {
                "status": "ok",
                "loose": loose_stage,
                "resume": resume_stage,
            }
        )
    except Exception as exc:
        row.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
    return row


def run_fresh_case(case, verbose=False):
    """Run a fresh tight solve from scratch."""
    name = case["name"]
    row = {"name": name}
    try:
        sol, data, fresh_inputs, sc = _run_logged_case(
            case["tight"], f"{name}-FRESH", verbose=verbose
        )
        row.update(
            {
                "status": "ok",
                "fresh": _make_stage_record("Fresh", sc, sol, data, fresh_inputs),
            }
        )
    except Exception as exc:
        row.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
    return row


#################################################################
#  Report writers
#################################################################
def write_report(path, title, rows, summary_keys=(), input_sections=(), log_sections=()):
    """Write a text report for resume-demo rows."""
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
    """Write combined loose/resume/fresh reports with stage summaries."""
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
        loose_stage = rrow.get("loose", {})
        resume_stage = rrow.get("resume", {})
        fresh_stage = frow.get("fresh", {})

        # Input sections
        for section_label, stage in [
            ("loose_inputs", loose_stage),
            ("resume_inputs", resume_stage),
            ("fresh_inputs", fresh_stage),
        ]:
            text = stage.get("inputs", "")
            if not text:
                continue
            lines.append("")
            lines.append(f"  {section_label}:")
            lines.extend([f"    {line}" for line in text.splitlines()])

        # Iteration log sections
        for section_label, stage in [
            ("loose_iteration_log", loose_stage),
            ("resume_iteration_log", resume_stage),
            ("fresh_iteration_log", fresh_stage),
        ]:
            text = stage.get("iteration_log", "")
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
                    loose_stage.get("tol"),
                    loose_stage.get("total_n", 0),
                    loose_stage.get("new_n", 0),
                    loose_stage.get("iters"),
                    loose_stage.get("solution", float("nan")),
                    loose_stage.get("half_width", float("nan")),
                    loose_stage.get("time", 0.0),
                ),
                (
                    "Resumed",
                    resume_stage.get("tol"),
                    resume_stage.get("total_n", 0),
                    resume_stage.get("new_n", 0),
                    resume_stage.get("iters"),
                    resume_stage.get("solution", float("nan")),
                    resume_stage.get("half_width", float("nan")),
                    resume_stage.get("time", 0.0),
                ),
                (
                    "Fresh",
                    fresh_stage.get("tol"),
                    fresh_stage.get("total_n", 0),
                    fresh_stage.get("new_n", 0),
                    fresh_stage.get("iters"),
                    fresh_stage.get("solution", float("nan")),
                    fresh_stage.get("half_width", float("nan")),
                    fresh_stage.get("time", 0.0),
                ),
            ]
            # Use target_rmse_tol as the column header when all rows report that name
            tol_names = {
                loose_stage.get("tol_name"),
                resume_stage.get("tol_name"),
                fresh_stage.get("tol_name"),
            } - {None}
            tol_header = "rmse_tol" if tol_names == {"target_rmse_tol"} else "abs_tol"
            summary_text = format_stage_summary(
                stage_rows, title=f"Stage summary of {name}", tol_header=tol_header
            ).strip()
            lines.append("")
            lines.extend([f"  {line}" for line in summary_text.splitlines()])

        for warning_line in collect_resume_fresh_warnings(
            name, resume_stage, fresh_stage
        ):
            print(f"  {warning_line}")
            lines.append(f"  {warning_line}")

        # Errors
        for row_obj in (rrow, frow):
            if "error" in row_obj:
                lines.append(f"  error: {row_obj['error']}")

        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
