"""Long-run resume check 
"""

from pathlib import Path
from qmcpy import (CubBayesNetG, CubMCCLTVec, CubMLMC, CubMLMCCont, CubMLQMC, CubMLQMCCont, CubQMCNetG, CubQMCLatticeG, 
    CubQMCRepStudentT, DigitalNetB2, CubQMCBayesLatticeG, FinancialOption, IIDStdUniform, Keister, Lattice)
from resume_util import (make_named_tol_builder, make_tol_case, print_stage_summary, run_fresh_case, run_resume_case,
                         stage_summary_rows_from_stage_records, stage_summary_tol_header, write_combined_report)


DEFAULT_SEED = 7
DEFAULT_CONT_SEED = 11
DEFAULT_DIMENSION = 2


def _build_cases(seed=DEFAULT_SEED, cont_seed=DEFAULT_CONT_SEED, dimension=DEFAULT_DIMENSION):
    # ------------------------------------------------------------------ #
    # Integrands
    # ------------------------------------------------------------------ #
    def iid_keister():
        return Keister(IIDStdUniform(dimension=dimension, seed=seed))

    def lattice_keister():
        return Keister(Lattice(dimension=dimension, seed=seed))

    def net_keister():
        return Keister(DigitalNetB2(dimension=dimension, seed=seed))

    def net_rep_keister_long():
        return Keister(DigitalNetB2(dimension=max(8, dimension), replications=16, seed=seed))

    def bayes_lattice_keister():
        return Keister(Lattice(dimension=dimension, seed=seed, order="RADICAL INVERSE"))

    # ------------------------------------------------------------------ #
    # Single-level IID/QMC solvers
    # ------------------------------------------------------------------ #
    clt_vec_builder = make_named_tol_builder(
        CubMCCLTVec, iid_keister, "abs_tol",
        n_init=2, n_limit=2**22,
    )
    lattice_builder = make_named_tol_builder(
        CubQMCLatticeG, lattice_keister, "abs_tol",
        n_init=2**8, n_limit=2**20,
    )
    net_builder = make_named_tol_builder(
        CubQMCNetG, net_keister, "abs_tol",
        n_init=2**8, n_limit=2**22,
    )
    rep_student_builder = make_named_tol_builder(
        CubQMCRepStudentT, net_rep_keister_long, "abs_tol",
        rel_tol=0, n_init=2**5, n_limit=2**22,
    )
    bayes_lattice_builder = make_named_tol_builder(
        CubQMCBayesLatticeG, bayes_lattice_keister, "abs_tol",
        n_init=2**2, n_limit=2**22,
    )
    bayes_net_builder = make_named_tol_builder(
        CubBayesNetG, net_keister, "abs_tol",
        n_init=2**2, n_limit=2**24,
    )

    # ------------------------------------------------------------------ #
    # MLMC / MLQMC
    # ------------------------------------------------------------------ #

    def iid_financial_option_4d():
        return FinancialOption(
            IIDStdUniform(dimension=4, seed=seed),
            start_price=300, strike_price=100,
        )

    mlmc_builder = make_named_tol_builder(
        CubMLMC, iid_financial_option_4d, "rmse_tol",
        n_limit=2**24,
    )

    def qmc_financial_option_4d():
        return FinancialOption(
            Lattice(replications=32, seed=seed, dimension=4),
            start_price=300, strike_price=100,
        )

    mlqmc_builder = make_named_tol_builder(
        CubMLQMC, qmc_financial_option_4d, "abs_tol",
        n_limit=2**24,
    )

    # ------------------------------------------------------------------ #
    # Cont solvers
    # ------------------------------------------------------------------ #

    def iid_financial_option_cont_large():
        return FinancialOption(
            IIDStdUniform(dimension=4, seed=cont_seed),
            start_price=300, strike_price=100,
        )

    def qmc_financial_option_cont_large():
        return FinancialOption(
            Lattice(replications=32, seed=cont_seed, dimension=4),
            start_price=300, strike_price=100,
        )

    mlmc_cont_builder = make_named_tol_builder(
        CubMLMCCont, iid_financial_option_cont_large, "rmse_tol",
        n_tols=1200, inflate=1.001, n_limit=2**24,
    )

    mlqmc_cont_builder = make_named_tol_builder(
        CubMLQMCCont, qmc_financial_option_cont_large, "abs_tol",
        n_tols=1200, inflate=1.001, n_limit=2**24,
    )

    return [
        make_tol_case("CubMCCLTVec",       clt_vec_builder,       loose_tol=5e-3,  tight_tol=2e-3),
        make_tol_case("CubQMCLatticeG",    lattice_builder,       loose_tol=1e-3,  tight_tol=1e-5),
        make_tol_case("CubQMCNetG",        net_builder,           loose_tol=1e-3,  tight_tol=1e-6),
        make_tol_case("CubQMCRepStudentT", rep_student_builder,   loose_tol=5e-3,  tight_tol=5e-4),
        make_tol_case("CubQMCBayesLatticeG", bayes_lattice_builder, loose_tol=1e-3, tight_tol=1e-6),
        make_tol_case("CubBayesNetG",      bayes_net_builder,     loose_tol=1e-3,  tight_tol=2e-6),
        make_tol_case("CubMLMC",           mlmc_builder,          loose_tol=2.0,   tight_tol=0.5),
        make_tol_case("CubMLQMC",          mlqmc_builder,         loose_tol=2.0,   tight_tol=0.5),
        make_tol_case("CubMLMCCont",       mlmc_cont_builder,     loose_tol=1.0,   tight_tol=0.5),
        make_tol_case("CubMLQMCCont",      mlqmc_cont_builder,    loose_tol=1.0,   tight_tol=0.5),
    ]


def main(verbose=False, seed=DEFAULT_SEED, cont_seed=DEFAULT_CONT_SEED, dimension=DEFAULT_DIMENSION):
    cases = _build_cases(seed=seed, cont_seed=cont_seed, dimension=dimension)
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    resume_rows = []
    fresh_rows = []
    for case in cases:
        resume_row = run_resume_case(case, verbose=verbose)
        fresh_row = run_fresh_case(case, verbose=verbose)
        resume_rows.append(resume_row)
        fresh_rows.append(fresh_row)
        if resume_row.get("status") == "ok" and fresh_row.get("status") == "ok":
            print_stage_summary(
                rows=stage_summary_rows_from_stage_records(
                    resume_row.get("loose"),
                    resume_row.get("resume"),
                    fresh_row.get("fresh"),
                ),
                title=f"Stage summary of {case['name']}",
                tol_header=stage_summary_tol_header(
                    resume_row.get("loose"),
                    resume_row.get("resume"),
                    fresh_row.get("fresh"),
                ),
            )

    combined_path = output_dir / "check_resume_long_summary.txt"
    write_combined_report(combined_path, "Stopping Criteria Long-Run Check: Resume vs Fresh (throttle=True)", resume_rows, fresh_rows)
    print(f"wrote: {combined_path}")


if __name__ == "__main__":
    main(verbose=False)
