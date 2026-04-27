from pathlib import Path

from qmcpy import (CubBayesLatticeG, CubBayesNetG, CubMCCLTVec, CubMLMC, CubMLMCCont, CubMLQMC,
    CubMLQMCCont, CubQMCNetG, CubQMCLatticeG, DigitalNetB2, FinancialOption, IIDStdUniform, Keister, Lattice)

DEFAULT_SEED = 7
DEFAULT_CONT_SEED = 11
DEFAULT_DIMENSION = 2


try:
    from .resume_util import (capture_integrate, enable_diagnostics, make_abs_tol_builder,
        make_case, make_named_tol_builder, make_tol_case, run_fresh_case, run_resume_case, write_combined_report)
except ImportError:
    from resume_util import (capture_integrate, enable_diagnostics, make_abs_tol_builder,
        make_case, make_named_tol_builder, make_tol_case, run_fresh_case, run_resume_case, write_combined_report)


def _build_cases(seed=7, cont_seed=11, dimension=2, loose_tol=0.2, tight_tol=0.05, rel_tol=0, n_init=2**8, n_limit=2**16):
    iid_keister = lambda dim=dimension: Keister(IIDStdUniform(dimension=dim, seed=seed))
    lattice_keister = lambda dim=dimension: Keister(Lattice(dimension=dim, seed=seed))
    net_keister = lambda dim=dimension: Keister(DigitalNetB2(dimension=dim, seed=seed))
    bayes_lattice_keister = lambda dim=dimension: Keister(Lattice(dimension=dim, seed=seed, order="RADICAL INVERSE"))

    def iid_financial_option():
        return FinancialOption(IIDStdUniform(dimension=dimension, seed=seed), start_price=30, strike_price=30)

    def iid_financial_option_cont():
        return FinancialOption(IIDStdUniform(dimension=dimension, seed=cont_seed), start_price=30, strike_price=30)

    def qmc_financial_option():
        return FinancialOption(Lattice(replications=32, seed=seed), start_price=30, strike_price=30)

    return [
        make_tol_case("CubMCCLTVec",
            make_abs_tol_builder(CubMCCLTVec, iid_keister, rel_tol=rel_tol, n_init=n_init, n_limit=n_limit),
            loose_tol, tight_tol),
        make_tol_case("CubQMCLatticeG",
            make_abs_tol_builder(CubQMCLatticeG, lattice_keister, rel_tol=rel_tol, n_init=n_init, n_limit=n_limit),
            1e-3, 1e-6),
        make_tol_case("CubQMCNetG",
            make_abs_tol_builder(CubQMCNetG, net_keister, rel_tol=rel_tol, n_init=n_init, n_limit=n_limit),
            1e-3, 1e-6),
        make_tol_case("CubBayesLatticeG",
            make_abs_tol_builder(CubBayesLatticeG, bayes_lattice_keister, rel_tol=rel_tol, n_init=2**5, n_limit=n_limit),
            loose_tol, tight_tol),
        make_tol_case("CubBayesNetG",
            make_abs_tol_builder(CubBayesNetG, net_keister, rel_tol=rel_tol, n_init=2**5, n_limit=n_limit),
            loose_tol, tight_tol),
        make_tol_case("CubMLMC",
            make_named_tol_builder(CubMLMC, iid_financial_option, "rmse_tol", n_limit=2**16),
            0.2, 0.1),
        make_tol_case("CubMLMCCont",
            make_named_tol_builder(CubMLMCCont, iid_financial_option_cont, "rmse_tol", n_limit=2**16),
            0.1, 0.05),
        make_tol_case("CubMLQMC",
            make_named_tol_builder(CubMLQMC, qmc_financial_option, "abs_tol", n_limit=2**18),
            0.2, 0.1),
        make_tol_case("CubMLQMCCont",
            make_named_tol_builder(CubMLQMCCont, qmc_financial_option, "abs_tol", n_limit=2**18),
            0.2, 0.1),
        # n_tols=20 produces >30 trace rows (20 outer tolerance loops × 2–3 inner MLMC iterations each)
        make_tol_case("CubMLMCContLong",
            make_named_tol_builder(CubMLMCCont, iid_financial_option_cont, "rmse_tol", n_tols=20, n_limit=2**18),
            0.1, 0.05),
    ]


def _demo_throttle_iterations(out_path, cont_seed=DEFAULT_CONT_SEED, dimension=DEFAULT_DIMENSION):
    """Run the long-iteration CubMLMCCont case (n_tols=20, >30 trace rows) with
    trace_throttle_iterations toggled on then off to show the difference.
    Output is appended to out_path."""
    def _make_solver():
        return CubMLMCCont(
            FinancialOption(
                IIDStdUniform(dimension=dimension, seed=cont_seed),
                start_price=30, strike_price=30,
            ),
            rmse_tol=0.05, n_tols=20, n_limit=2**18,
        )

    lines = ["", "=" * 60, "trace_throttle_iterations demo (CubMLMCCont, n_tols=20)", "=" * 60]
    for throttle in (True, False):
        sc = _make_solver()
        enable_diagnostics(sc, "CubMLMCCont-long", throttle_iterations=throttle)
        (solution, data), log = capture_integrate(sc)
        n_rows = sum(1 for line in log.splitlines() if line.startswith(("ITER", "RESUME")))
        total_iters = getattr(data, "_iter_count", "?")
        lines.append(f"\n--- sc.trace_throttle_iterations = {throttle} ---")
        lines.append(f"    {n_rows} log rows printed, {total_iters} total iterations")
        lines.append(log)

    with open(out_path, "a") as f:
        f.write("\n".join(lines) + "\n")


def main(throttle_iterations=True, seed=7, cont_seed=11, dimension=2):
    # Fix all demo sampler seeds here so the reported solution estimates are reproducible.
    cases = _build_cases(seed=seed, cont_seed=cont_seed, dimension=dimension)
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    resume_rows = [run_resume_case(case, throttle_iterations=throttle_iterations) for case in cases]
    fresh_rows = [run_fresh_case(case, throttle_iterations=throttle_iterations) for case in cases]

    combined_path = output_dir / "check_resume_summary.txt"
    write_combined_report(combined_path, "Stopping Criteria Check: Resume vs Fresh", resume_rows, fresh_rows)
    print(f"wrote: {combined_path}")
    _demo_throttle_iterations(combined_path, cont_seed=cont_seed, dimension=dimension)


if __name__ == "__main__":
    main(throttle_iterations=False, seed=DEFAULT_SEED, cont_seed=DEFAULT_CONT_SEED, dimension=DEFAULT_DIMENSION)
