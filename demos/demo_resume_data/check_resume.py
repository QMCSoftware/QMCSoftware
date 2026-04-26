from pathlib import Path

from qmcpy import (
    CubBayesLatticeG,
    CubBayesNetG,
    CubMCCLTVec,
    CubMCML,
    CubMCMLCont,
    CubQMCML,
    CubQMCMLCont,
    CubQMCNetG,
    CubQMCLatticeG,
    DigitalNetB2,
    FinancialOption,
    IIDStdUniform,
    Keister,
    Lattice,
)


try:
    from .resume_util import (
        make_abs_tol_builder,
        make_case,
        make_named_tol_builder,
        make_tol_case,
        run_fresh_case,
        run_resume_case,
        write_combined_report,
    )
except ImportError:
    from resume_util import (
        make_abs_tol_builder,
        make_case,
        make_named_tol_builder,
        make_tol_case,
        run_fresh_case,
        run_resume_case,
        write_combined_report,
    )


def _build_cases(seed=7, dimension=2, loose_tol=0.2, tight_tol=0.05, rel_tol=0, n_init=2**8, n_limit=2**16):
    iid_keister = lambda dim=dimension: Keister(
        IIDStdUniform(dimension=dim, seed=seed)
    )
    lattice_keister = lambda dim=dimension: Keister(Lattice(dimension=dim, seed=seed))
    net_keister = lambda dim=dimension: Keister(DigitalNetB2(dimension=dim, seed=seed))
    bayes_lattice_keister = lambda dim=dimension: Keister(
        Lattice(dimension=dim, seed=seed, order="RADICAL INVERSE")
    )

    def iid_financial_option():
        return FinancialOption(
            IIDStdUniform(dimension=dimension, seed=seed),
            start_price=30,
            strike_price=30,
        )

    def qmc_financial_option():
        return FinancialOption(
            Lattice(replications=32, seed=seed),
            start_price=30,
            strike_price=30,
        )

    return [
        make_tol_case(
            "CubMCCLTVec",
            make_abs_tol_builder(
                CubMCCLTVec,
                iid_keister,
                rel_tol=rel_tol,
                n_init=n_init,
                n_limit=n_limit,
            ),
            loose_tol,
            tight_tol,
        ),
        make_tol_case(
            "CubQMCLatticeG",
            make_abs_tol_builder(
                CubQMCLatticeG,
                lattice_keister,
                rel_tol=rel_tol,
                n_init=n_init,
                n_limit=n_limit,
            ),
            1e-3,
            1e-6,
        ),
        make_tol_case(
            "CubQMCNetG",
            make_abs_tol_builder(
                CubQMCNetG,
                net_keister,
                rel_tol=rel_tol,
                n_init=n_init,
                n_limit=n_limit,
            ),
            1e-3,
            1e-6,
        ),
        make_tol_case(
            "CubBayesLatticeG",
            make_abs_tol_builder(
                CubBayesLatticeG,
                bayes_lattice_keister,
                rel_tol=rel_tol,
                n_init=2**5,
                n_limit=n_limit,
            ),
            loose_tol,
            tight_tol,
        ),
        make_tol_case(
            "CubBayesNetG",
            make_abs_tol_builder(
                CubBayesNetG,
                net_keister,
                rel_tol=rel_tol,
                n_init=2**5,
                n_limit=n_limit,
            ),
            loose_tol,
            tight_tol,
        ),
        make_tol_case(
            "CubMCML",
            make_named_tol_builder(
                CubMCML, iid_financial_option, "rmse_tol", n_limit=2**18
            ),
            0.05,
            0.04,
        ),
        make_tol_case(
            "CubMCMLCont",
            make_named_tol_builder(
                CubMCMLCont, iid_financial_option, "rmse_tol", n_limit=2**16
            ),
            0.06,
            0.05,
        ),
        make_tol_case(
            "CubQMCML",
            make_named_tol_builder(
                CubQMCML, qmc_financial_option, "abs_tol", n_limit=2**22
            ),
            0.025,
            0.02,
        ),
        make_tol_case(
            "CubQMCMLCont",
            make_named_tol_builder(
                CubQMCMLCont, qmc_financial_option, "abs_tol", n_limit=2**22
            ),
            0.025,
            0.02,
        ),
    ]


def main(throttle_iterations=True):
    cases = _build_cases()
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    resume_rows = [
        run_resume_case(case, throttle_iterations=throttle_iterations)
        for case in cases
    ]
    fresh_rows = [
        run_fresh_case(case, throttle_iterations=throttle_iterations)
        for case in cases
    ]

    combined_path = output_dir / "check_resume_summary.txt"
    write_combined_report(
        combined_path,
        "Stopping Criteria Check: Resume vs Fresh",
        resume_rows,
        fresh_rows,
    )
    print(f"wrote: {combined_path}")


if __name__ == "__main__":
    main(throttle_iterations=False)
