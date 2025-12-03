import glob
import os
import re
import time
from pathlib import Path

import parsl
import parsl as pl
from parsl import bash_app
from parsl.configs.htex_local import config

# Import runtime estimates for load balancing (LPT scheduling)
try:
    from test_runtimes import optimal_schedule, get_runtime
    HAS_RUNTIME_ESTIMATES = True
except ImportError:
    HAS_RUNTIME_ESTIMATES = False

# ---------------------------------------------------------------------------
# SINGLE SOURCE OF TRUTH FOR SKIPS
#
# These are the NOTEBOOKS we explicitly skip (with reasons).
# tb_*.py modules are derived from these paths automatically, so there is
# NO separate SKIP_MODULES list anymore.
# ---------------------------------------------------------------------------
SKIPPED_NOTEBOOKS = {
    "demos/DAKOTA_Genz/dakota_genz.ipynb":
        "requires large manual file / heavy memory use",
    "demos/talk_paper_demos/Argonne_Talk_2023_May/Argonne_2023_Talk_Figures.ipynb":
        "heavy LaTeX + many figures; skipped in booktests_no_docker",
    "demos/talk_paper_demos/MCQMC2022_Article_Figures/MCQMC2022_Article_Figures.ipynb":
        "MCQMC 2022 article figures; not run in CI",
    "demos/talk_paper_demos/ProbFailureSorokinRao/prob_failure_gp_ci.ipynb":
        "prob_failure_gp_ci talk demo; heavy GP / prob. failure example",
    "demos/talk_paper_demos/Purdue_Talk_2023_March/Purdue_Talk_Figures.ipynb":
        "Purdue 2023 talk figures; not a CI booktest target",
    "demos/talk_paper_demos/parsel_fest_2025/01_sequential.ipynb":
        "helper notebook for parsl_fest_2025; not a standalone booktest",
    "demos/talk_paper_demos/parsel_fest_2025/02_parallel.ipynb":
        "helper notebook for parsl_fest_2025; not a standalone booktest",
    "demos/talk_paper_demos/parsel_fest_2025/03_visualize_speedup.ipynb":
        "helper notebook for parsl_fest_2025; not a standalone booktest",
    "demos/talk_paper_demos/pydata_chi_2023.ipynb":
        "PyData Chicago 2023 talk notebook; demo-only",
}
# ---------------------------------------------------------------------------


def _derive_skipped_modules_from_notebooks():
    """
    Derive tb_*.py module names from SKIPPED_NOTEBOOKS keys.

    For a notebook path with filename base 'foo-bar.ipynb', the corresponding
    tb module is 'tb_foo_bar', mirroring the generate_test.py naming logic.
    """
    skipped_modules = set()
    for rel_path in SKIPPED_NOTEBOOKS.keys():
        base = Path(rel_path).stem                 # e.g. 'dakota_genz'
        test_base = re.sub(r"[-.]", "_", base)     # e.g. 'dakota_genz' or 'foo_bar'
        skipped_modules.add(f"tb_{test_base}")     # e.g. 'tb_dakota_genz'
    return skipped_modules


def print_notebook_booktest_coverage():
    """
    Print notebook ↔ booktest coverage and explicit skip list.

    This mirrors the Makefile `check_booktests` logic:
    - lists explicitly skipped demos/*.ipynb and reasons
    - counts total notebooks, skipped, with tests, and missing tests
    - reports how many notebooks will have booktests run
    """
    # parsl_test_runner.py lives in test/booktests → repo root is two levels up
    repo_root = Path(__file__).resolve().parents[2]
    demos_root = repo_root / "demos"
    booktests_root = repo_root / "test" / "booktests"

    print("Checking notebook ↔ booktest coverage...")

    total_notebooks = 0
    skipped_explicit = 0
    with_tests = 0
    missing_tests = 0

    for nb_path in sorted(demos_root.rglob("*.ipynb")):
        rel = nb_path.relative_to(repo_root).as_posix()
        total_notebooks += 1

        base = nb_path.stem
        test_base = re.sub(r"[-.]", "_", base)
        tb_path = booktests_root / f"tb_{test_base}.py"

        # Explicit skip list with reasons
        if rel in SKIPPED_NOTEBOOKS:
            skipped_explicit += 1
            reason = SKIPPED_NOTEBOOKS[rel]
            print(f"    Skipping {rel} ({reason})")
            continue

        # Check if a tb_*.py exists
        if tb_path.exists():
            with_tests += 1
        else:
            missing_tests += 1
            print(
                f"    Missing test for: {rel} -> "
                f"Expected: test/booktests/tb_{test_base}.py"
            )

    print(f"Total notebooks found:        {total_notebooks}")
    print(f"Total skipped explicitly (from .py creation):     {skipped_explicit}")
    print(f"Total notebooks WITH tests:   {with_tests}")
    print(f"Total notebooks MISSING tests:{missing_tests}")
    print(f"Booktests will be run for {with_tests} notebooks.\n")


@bash_app
def run_single_test(test_module, stdout="test_output.txt", stderr="test_error.txt"):
    """Run a single tb_*.py unittest module via bash."""
    return f"""
    PYTHONWARNINGS="ignore::UserWarning,ignore::DeprecationWarning,ignore::FutureWarning,ignore::ImportWarning" \
    python -m unittest {test_module}
    """


def execute_parallel_tests():
    """Execute all tb_*.py booktests in parallel using Parsl."""
    start_time = time.time()

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Discover test modules from CLI args or tb_*.py files
    # ------------------------------------------------------------------
    import sys

    test_modules = []
    for arg in sys.argv[1:]:
        if arg.startswith("-"):
            break
        test_modules.append(arg)

    if not test_modules:
        test_files = glob.glob("tb_*.py")
        test_modules = [os.path.basename(f).replace(".py", "") for f in test_files]

    # Deduplicate while preserving order
    seen = set()
    unique_test_modules = []
    for module in test_modules:
        if module not in seen:
            seen.add(module)
            unique_test_modules.append(module)
    if len(unique_test_modules) < len(test_modules):
        print(f"Note: Removed {len(test_modules) - len(unique_test_modules)} duplicate test module(s)")
    test_modules = unique_test_modules

    # ------------------------------------------------------------------
    # Drop modules derived from SKIPPED_NOTEBOOKS
    # ------------------------------------------------------------------
    skip_modules = _derive_skipped_modules_from_notebooks()
    test_modules = [m for m in test_modules if m not in skip_modules]
    # ------------------------------------------------------------------
    # LPT scheduling using runtime estimates (optional)
    # ------------------------------------------------------------------
    if HAS_RUNTIME_ESTIMATES and test_modules:
        try:
            if getattr(config, "executors", None) and hasattr(config.executors[0], "max_workers"):
                num_workers = config.executors[0].max_workers
            elif hasattr(config, "max_workers"):
                num_workers = config.max_workers
            else:
                num_workers = 1
        except Exception:
            num_workers = 1

        if num_workers < 1:
            num_workers = 1

        test_modules, assignments, est_makespan = optimal_schedule(test_modules, num_workers)
        total_seq = sum(get_runtime(m) for m in test_modules)

        # Fallback if est_makespan is 0 / None
        if not est_makespan or est_makespan <= 0:
            longest = max((get_runtime(m) for m in test_modules), default=0.0)
            if longest > 0 and total_seq > 0:
                est_makespan = max(longest, total_seq / num_workers)
            else:
                est_makespan = total_seq

        if est_makespan > 0 and total_seq > 0:
            est_speedup = total_seq / est_makespan
            est_speedup_str = f"{est_speedup:.2f}x"
        else:
            est_speedup_str = "1.00x"

        print(f"Applied optimal LPT scheduling for {num_workers} workers")
        print(f"Estimated makespan: {est_makespan:.1f}s (speedup: {est_speedup_str})")

    print(f"Found {len(test_modules)} test modules to execute in parallel...")

    # ------------------------------------------------------------------
    # Submit all jobs
    # ------------------------------------------------------------------
    futures = []
    for i, module in enumerate(test_modules):
        future = run_single_test(
            module,
            stdout=f"logs/test_{i}_{module}.out",
            stderr=f"logs/test_{i}_{module}.err",
        )
        futures.append((module, future, i))

    print("All tests submitted to Parsl executor...")

    # ------------------------------------------------------------------
    # Wait for completion and collect results
    # ------------------------------------------------------------------
    results = []
    completed = 0
    processed_modules = set()

    for module, future, index in futures:
        if module in processed_modules:
            print(f"WARNING: Module {module} already processed - skipping duplicate!")
            continue
        processed_modules.add(module)

        was_retried = False

        try:
            future.result()
        except Exception as e:
            error_str = str(e)
            if "unix exit code 5" in error_str:
                # NO TESTS RAN (all skipped) is fine
                pass
            else:
                print(f"Test {module} failed once with error: {e}. Retrying...")
                try:
                    retry_future = run_single_test(
                        module,
                        stdout=f"logs/test_{index}_{module}_retry.out",
                        stderr=f"logs/test_{index}_{module}_retry.err",
                    )
                    retry_future.result()
                    was_retried = True
                except Exception as e2:
                    if "unix exit code 5" in str(e2):
                        was_retried = True
                    else:
                        results.append((module, f"FAILED after retry: {e2}", 0))
                        status = "FAILED"
                        completed += 1
                        print(f"[{completed}/{len(futures)}] {module}: {status}")
                        continue

        # Decide which output file to read
        if was_retried:
            output_file = f"logs/test_{index}_{module}_retry.out"
            error_file = f"logs/test_{index}_{module}_retry.err"
        else:
            output_file = f"logs/test_{index}_{module}.out"
            error_file = f"logs/test_{index}_{module}.err"

        skip_count = 0
        error_content = ""
        if os.path.exists(error_file):
            with open(error_file, "r") as f:
                error_content = f.read()
                if "NO TESTS RAN" in error_content and "skipped=" in error_content:
                    match = re.search(r"NO TESTS RAN \(skipped=(\d+)\)", error_content)
                    if match:
                        skip_count = int(match.group(1))

        output_content = ""
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                output_content = f.read()
                if skip_count == 0:
                    match = re.search(r"OK \(skipped=(\d+)\)", output_content)
                    if match:
                        skip_count = int(match.group(1))
                    elif "skipped" in output_content.lower():
                        skip_count = output_content.lower().count("skipped")

        results.append((module, "PASSED", skip_count))
        status = f"PASSED (skipped={skip_count})" if skip_count > 0 else "PASSED"
        completed += 1
        print(f"[{completed}/{len(futures)}] {module}: {status}")


        # Echo memory + time if we can parse it from BaseNotebookTest.tearDown()
        # We only care about the "Memory used: X GB.  Test time: Y s" line, e.g.:
        #     [i/n] Memory used: 0.16 GB.  Test time: 384.70 s
        if output_content:
            mem_match = re.search(
                r"Memory used:\s+(?P<mem>[\d\.]+)\s+GB\.\s+Test time:\s+(?P<time>[\d\.]+)\s+s",
                output_content,
            )
            if mem_match:
                mem_used = mem_match.group("mem")
                test_time_val = mem_match.group("time")
                # Use module name as the identifier; ordering may differ from sequential,
                # but we still get per-test memory + time in the parallel log.
                print(
                    f"{module} ...     Memory used: {mem_used} GB.  Test time: {test_time_val} s"
                )
                print("ok")
                
        # Clean up logs
        try:
            if os.path.exists(output_file):
                os.remove(output_file)
            if os.path.exists(error_file):
                os.remove(error_file)
        except Exception:
            pass

    execution_time = time.time() - start_time
    return results, execution_time


def generate_summary_report(results, execution_time=0.0):
    """Generate a unittest-style summary report for the Parsl run."""
    total_modules = len(results)
    passed_modules = sum(1 for _, status, _ in results if status == "PASSED")
    failed_modules = total_modules - passed_modules
    total_skipped = sum(skip_count for _, status, skip_count in results if status == "PASSED")

    status_line = ""
    for _, status, skip_count in results:
        if status == "PASSED":
            if skip_count > 0:
                status_line += "s" * skip_count + "."
            else:
                status_line += "."
        else:
            status_line += "F"

    print(status_line)
    print("-" * 70)
    print(f"Ran {total_modules} test modules in {execution_time:.3f}s\n")

    if failed_modules == 0:
        if total_skipped > 0:
            print(f"OK (skipped={total_skipped})")
        else:
            print("OK")
    else:
        if total_skipped > 0:
            print(f"FAILED (failures={failed_modules}, skipped={total_skipped})")
        else:
            print(f"FAILED (failures={failed_modules})")
        print()
        print("FAILURES:")
        print("=" * 70)
        for module, status, _ in results:
            if status != "PASSED":
                print(f"FAIL: {module}")
                print("-" * 70)
                print(f"Error: {status}")
                print()


def main():
    """
    CLI entry point for Parsl booktest execution.

    - Supports existing usage from Makefile:
        python parsl_test_runner.py tb_foo tb_bar -v --failfast

    - Supports notebook / demo usage:
        python parsl_test_runner.py --workers 4 --print-coverage

    Coverage printing (skipped notebooks, totals, etc.) only happens when
    --print-coverage is passed, so it does NOT duplicate Makefile output.
    """
    import argparse
    import parsl as pl
    from parsl.configs.htex_local import config

    parser = argparse.ArgumentParser(
        description="Run QMCPy booktests in parallel using Parsl."
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of Parsl workers (htex_local max_workers).",
    )

    parser.add_argument(
        "--print-coverage",
        action="store_true",
        help="Print notebook ↔ booktest coverage summary before running tests.",
    )

    # Keep unknown args (tb_*.py, -v, --failfast) for unittest
    args, unknown = parser.parse_known_args()

    # Optional coverage block (used from demo notebook)
    if args.print_coverage:
        print_notebook_booktest_coverage()

    # Configure workers in Parsl config (if provided)
    if args.workers is not None:
        w = max(1, args.workers)
        setattr(config, "max_workers", w)
        if getattr(config, "executors", None):
            for ex in config.executors:
                if hasattr(ex, "max_workers"):
                    ex.max_workers = w
        print(f"Configuring Parsl with max_workers={w}")
    else:
        w = getattr(config, "max_workers", None)
        if w is not None:
            print(f"Using existing Parsl config max_workers={w}")
        else:
            print("Using Parsl default worker configuration")

    # Load Parsl once for this process
    try:
        pl.load(config)
        print("Parsl configuration loaded successfully.")
    except AssertionError as ae:
        msg = str(ae)
        if "Already exists!" in msg:
            print("Parsl already configured; reusing existing DFK.")
        else:
            print(f"Error loading Parsl configuration: {ae}")
            return
    except Exception as e:
        print(f"Error loading Parsl configuration: {e}")
        return

    # Run the tests
    try:
        results, execution_time = execute_parallel_tests()
        generate_summary_report(results, execution_time)
    except Exception as e:
        print(f"Error during parallel execution: {e}")
    finally:
        try:
            dfk = pl.dfk()
            dfk.cleanup()
            dfk.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
