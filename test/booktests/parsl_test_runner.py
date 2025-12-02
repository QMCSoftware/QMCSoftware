import glob
import os
import re
import subprocess
import signal
import time
from pathlib import Path

import parsl
import parsl as pl
from parsl import bash_app
from parsl.configs.htex_local import config

# Import runtime estimates for load balancing (LPT scheduling)
try:
    from test_runtimes import sort_by_runtime, get_runtime, print_schedule, optimal_schedule
    HAS_RUNTIME_ESTIMATES = True
except ImportError:
    HAS_RUNTIME_ESTIMATES = False


# ---------------------------------------------------------------------------
# EXPLICIT MODULE SKIP LIST
#
# These are tb_*.py modules that we DO NOT want Parsl to execute in parallel.
# Typically they correspond to notebooks that are:
#   - heavy LaTeX / many figures,
#   - orchestration-only demos,
#   - or otherwise excluded from CI booktests.
#
# To skip a new tb_*.py module from Parsl runs, ADD ITS MODULE NAME HERE.
# Example: to skip test/booktests/tb_new_heavy_demo.py, add "tb_new_heavy_demo".
# ---------------------------------------------------------------------------
SKIP_MODULES = {
    "tb_dakota_genz",
    "tb_Argonne_2023_Talk_Figures",
    "tb_MCQMC2022_Article_Figures",
    "tb_prob_failure_gp_ci",
    "tb_Purdue_Talk_Figures",
    "tb_pydata_chi_2023",
}
# ---------------------------------------------------------------------------


@bash_app
def run_single_test(test_module, stdout='test_output.txt', stderr='test_error.txt'):
    """Run a single tb_*.py unittest module via bash."""
    # We run `python -m unittest <module>` where <module> is e.g. tb_qmcpy_intro
    # NOTE: no .py extension here.
    return f"""
    PYTHONWARNINGS="ignore::UserWarning,ignore::DeprecationWarning,ignore::FutureWarning,ignore::ImportWarning" \
    python -m unittest {test_module}
    """


def _kill_interchange_processes(retries=3, delay=0.5):
    """Best-effort cleanup of any lingering Parsl 'interchange.py' processes."""
    for attempt in range(retries):
        # Try pkill first
        try:
            subprocess.run(['pkill', '-f', 'interchange.py'], check=False)
        except Exception:
            pass

        # Try pgrep -> kill
        try:
            p = subprocess.run(
                ['pgrep', '-f', 'interchange.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            if p.stdout:
                for line in p.stdout.splitlines():
                    try:
                        pid = int(line.strip())
                        try:
                            os.kill(pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                    except Exception:
                        pass
        except Exception:
            pass

        time.sleep(delay)

        # Force kill remaining
        try:
            p = subprocess.run(
                ['pgrep', '-f', 'interchange.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            if p.stdout:
                for line in p.stdout.splitlines():
                    try:
                        pid = int(line.strip())
                        try:
                            os.kill(pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    except Exception:
                        pass
        except Exception:
            pass

        # If none found, we’re done
        p_check = subprocess.run(
            ['pgrep', '-f', 'interchange.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        if not p_check.stdout:
            return True

        time.sleep(delay)

    # Final check
    final_check = subprocess.run(
        ['pgrep', '-f', 'interchange.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return not bool(final_check.stdout)


def reload_parsl_config(max_workers=None, wait=1.0):
    """Safely reload Parsl with an updated config.

    - Clean up any existing DFK (DataFlowKernel).
    - Kill lingering interchange processes.
    - Optionally set executor max_workers.
    - Load htex_local config fresh.
    """
    # Attempt graceful DFK cleanup if present
    try:
        dfk = pl.dfk()
        if dfk is not None:
            try:
                dfk.cleanup()
            except Exception:
                pass
            try:
                getattr(dfk, "shutdown", lambda: None)()
            except Exception:
                pass
    except Exception:
        # No active DFK or incompatible API — continue
        pass

    # Clear Parsl global state
    try:
        pl.clear()
    except Exception:
        pass

    # Ensure any leftover interchange processes are dead
    try:
        _kill_interchange_processes()
    except Exception:
        pass

    # Apply requested worker count
    if max_workers is not None:
        # Set top-level max_workers for logging
        config.max_workers = max_workers
        # Also set max_workers on each executor (this is what htex actually uses)
        if getattr(config, "executors", None):
            for ex in config.executors:
                if hasattr(ex, "max_workers"):
                    ex.max_workers = max_workers

    # Load the (possibly updated) config, with a couple of retries
    last_exc = None
    for _ in range(3):
        try:
            pl.load(config)
            break
        except AssertionError as ae:
            last_exc = ae
            try:
                _kill_interchange_processes()
            except Exception:
                pass
            time.sleep(0.5)
        except Exception as e:
            last_exc = e
            time.sleep(0.5)
    else:
        # Failed after retries
        raise last_exc

    time.sleep(wait)

    try:
        mw = getattr(config, "max_workers", None)
        print(f"Parsl loaded (max_workers={mw})")
        if getattr(pl, "config", None) is not None:
            try:
                exe_names = [type(e).__name__ for e in pl.config.executors]
                print("Executors:", exe_names)
            except Exception:
                pass
    except Exception:
        pass

    return pl


def execute_parallel_tests():
    """Execute all tb_*.py booktests in parallel using Parsl."""
    start_time = time.time()

    # Ensure logs directory exists
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Discover test modules from CLI args or from tb_*.py files.
    # ------------------------------------------------------------------
    import sys

    test_modules = []
    for arg in sys.argv[1:]:
        if arg.startswith("-"):
            break
        test_modules.append(arg)

    if not test_modules:
        # No explicit modules passed; run all tb_*.py files
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
    # APPLY MODULE-LEVEL SKIPS (SKIP_MODULES)
    #
    # This is where we keep tb_*.py modules that should NOT run in Parsl,
    # even though they may exist and contain only skipped tests.
    #
    # To add new skips, edit SKIP_MODULES above.
    # ------------------------------------------------------------------
    before_skip = len(test_modules)
    test_modules = [m for m in test_modules if m not in SKIP_MODULES]
    after_skip = len(test_modules)
    if after_skip != before_skip:
        print(f"Skipping {before_skip - after_skip} module(s) from Parsl runs via SKIP_MODULES.")

    # ------------------------------------------------------------------
    # Optional: LPT-based scheduling using runtime estimates.
    # ------------------------------------------------------------------
    if HAS_RUNTIME_ESTIMATES:
        # First try to read worker count from environment (source of truth)
        env_workers = os.environ.get("QMC_PARSL_WORKERS")
        if env_workers is not None:
            try:
                num_workers = int(env_workers)
            except ValueError:
                num_workers = None
        else:
            num_workers = None

        # Fallback: try to get it from Parsl config if env is missing/invalid
        if num_workers is None:
            try:
                num_workers = config.executors[0].max_workers if config.executors else 8
            except Exception:
                num_workers = 8

        # Use optimal bin-packing schedule
        test_modules, assignments, est_makespan = optimal_schedule(test_modules, num_workers)
        total_seq = sum(get_runtime(m) for m in test_modules)
        print(f"Applied optimal LPT scheduling for {num_workers} workers")
        print(f"Estimated makespan: {est_makespan:.1f}s (speedup: {total_seq / est_makespan:.2f}x)")

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
    processed_modules = set()  # Track which modules we've already processed

    for module, future, index in futures:
        if module in processed_modules:
            print(f"WARNING: Module {module} already processed - skipping duplicate!")
            continue
        processed_modules.add(module)

        was_retried = False

        try:
            future.result()  # Wait for completion
        except Exception as e:
            error_str = str(e)
            if "unix exit code 5" in error_str:
                # Exit code 5 => "NO TESTS RAN", which is OK for skip-only modules
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

        # Count skipped tests
        skip_count = 0
        if os.path.exists(error_file):
            with open(error_file, "r") as f:
                error_content = f.read()
                if "NO TESTS RAN" in error_content and "skipped=" in error_content:
                    match = re.search(r"NO TESTS RAN \(skipped=(\d+)\)", error_content)
                    if match:
                        skip_count = int(match.group(1))

        output_content = ""
        if os.path.exists(output_file) and skip_count == 0:
            with open(output_file, "r") as f:
                output_content = f.read()
                match = re.search(r"OK \(skipped=(\d+)\)", output_content)
                if match:
                    skip_count = int(match.group(1))
                elif "skipped" in output_content.lower():
                    skip_count = output_content.lower().count("skipped")

        results.append((module, "PASSED", skip_count))
        status = f"PASSED (skipped={skip_count})" if skip_count > 0 else "PASSED"
        completed += 1
        print(f"[{completed}/{len(futures)}] {module}: {status}")

        # Extract memory/time from output if present
        test_name = module
        test_case = ""
        mem_used = ""
        test_time_val = ""
        ok_status = "ok"

        match = re.search(
            r"(?P<test_case>[\w\.]+)\s+\.\.\.\s+Memory used:\s+(?P<mem>[\d\.]+)\s+GB\.\s+Test time:\s+(?P<time>[\d\.]+)\s+s",
            output_content,
        )
        if match:
            test_case = match.group("test_case")
            mem_used = match.group("mem")
            test_time_val = match.group("time")
            print(
                f"{test_name} ({test_case}) ...     Memory used: {mem_used} GB.  "
                f"Test time: {test_time_val} s\n{ok_status}"
            )

        # Clean up log files
        try:
            if os.path.exists(output_file):
                os.remove(output_file)
            if os.path.exists(error_file):
                os.remove(error_file)
        except Exception:
            pass

    end_time = time.time()
    execution_time = end_time - start_time

    return results, execution_time


def generate_summary_report(results, execution_time=0.0):
    """Generate a unittest-style summary report for the Parsl run."""
    total_modules = len(results)
    passed_modules = sum(1 for _, status, _ in results if status == "PASSED")
    failed_modules = total_modules - passed_modules
    total_skipped = sum(skip_count for _, status, skip_count in results if status == "PASSED")

    # Build unittest-style status line
    status_line = ""
    for module, status, skip_count in results:
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
    """Main entry point for Parsl booktest execution.

    Worker count is controlled by the environment variable QMC_PARSL_WORKERS.
    If not set or invalid, we default to 8.
    """
    # Read desired worker count from environment
    desired_workers = os.environ.get("QMC_PARSL_WORKERS")
    if desired_workers is not None:
        try:
            desired_workers = int(desired_workers)
        except ValueError:
            print(f"WARNING: Invalid QMC_PARSL_WORKERS={desired_workers!r}, falling back to default.")
            desired_workers = None

    if desired_workers is None:
        desired_workers = 8

    # Configure Parsl with this worker count
    try:
        reload_parsl_config(max_workers=desired_workers)
    except Exception as e:
        print(f"Error loading Parsl configuration: {e}")
        return None

    print(f"Parsl configuration loaded successfully (max_workers={desired_workers}).")

    try:
        results, execution_time = execute_parallel_tests()
        generate_summary_report(results, execution_time)
        return results
    except Exception as e:
        print(f"Error during parallel execution: {e}")
        return None
    finally:
        # Clean up Parsl
        try:
            parsl.clear()
        except Exception:
            pass
        try:
            dfk = parsl.dfk()
            dfk.cleanup()
            dfk.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
