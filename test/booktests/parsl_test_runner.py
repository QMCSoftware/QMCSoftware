import parsl
from parsl import bash_app
import glob
import os
import re
import time
from pathlib import Path
import time
import parsl as pl
from parsl.configs.htex_local import config

# Import runtime estimates for load balancing
try:
    from test_runtimes import sort_by_runtime, get_runtime, print_schedule, optimal_schedule
    HAS_RUNTIME_ESTIMATES = True
except ImportError:
    HAS_RUNTIME_ESTIMATES = False

@bash_app
def run_single_test(test_file, stdout='test_output.txt', stderr='test_error.txt'):
    """Run a single test file using bash"""
    return f"""
    PYTHONWARNINGS="ignore::UserWarning,ignore::DeprecationWarning,ignore::FutureWarning,ignore::ImportWarning" python -m unittest {test_file}
    """

def execute_parallel_tests():
    """Execute all testbook tests in parallel using Parsl"""
    import time
    start_time = time.time()
    
    # Ensure logs directory exists
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    
    import sys
    # Only treat arguments before the first option (starting with '-') as test modules
    test_modules = []
    for arg in sys.argv[1:]:
        if arg.startswith('-'):
            break
        test_modules.append(arg)
    if not test_modules:
        test_files = glob.glob('tb_*.py')
        test_modules = [os.path.basename(f).replace('.py', '') for f in test_files]
    
    # De-duplicate test modules while preserving order
    seen = set()
    unique_test_modules = []
    for module in test_modules:
        if module not in seen:
            seen.add(module)
            unique_test_modules.append(module)
    if len(unique_test_modules) < len(test_modules):
        print(f"Note: Removed {len(test_modules) - len(unique_test_modules)} duplicate test module(s)")
    test_modules = unique_test_modules

    # Apply optimal scheduling for better load balancing
    if HAS_RUNTIME_ESTIMATES:
        # Try to get the number of workers from Parsl config
        try:
            num_workers = config.executors[0].max_workers if config.executors else 8
        except:
            num_workers = 8
        
        # Use optimal bin-packing schedule
        test_modules, assignments, est_makespan = optimal_schedule(test_modules, num_workers)
        total_seq = sum(get_runtime(m) for m in test_modules)
        print(f"Applied optimal LPT scheduling for {num_workers} workers")
        print(f"Estimated makespan: {est_makespan:.1f}s (speedup: {total_seq/est_makespan:.2f}x)")
    
    print(f"Found {len(test_modules)} test modules to execute in parallel...")
    
    # Submit all jobs
    futures = []
    for i, module in enumerate(test_modules):
        future = run_single_test(
            module,
            stdout=f'logs/test_{i}_{module}.out',
            stderr=f'logs/test_{i}_{module}.err'
        )
        futures.append((module, future, i))
    
    print("All tests submitted to Parsl executor...")
    
    # Wait for completion and collect results
    results = []
    completed = 0
    processed_modules = set()  # Track which modules we've already processed
    for module, future, index in futures:
        if module in processed_modules:
            print(f"WARNING: Module {module} already processed - skipping duplicate!")
            continue
        processed_modules.add(module)
        was_retried = False
        exit_code_5_first_attempt = False
        try:
            future.result()  # Wait for completion
        except Exception as e:
            # Check if this is exit code 5 (NO TESTS RAN - all skipped)
            # Exit code 5 means all tests were skipped, which is success, not failure
            error_str = str(e)
            if 'unix exit code 5' in error_str:
                exit_code_5_first_attempt = True
                # Don't retry - exit code 5 is expected for skipped tests
            else:
                print(f"Test {module} failed once with error: {e}. Retrying...")
                try:
                    # Resubmit the test for retry
                    # Use _retry suffix for log files to preserve original failed test logs
                    # for debugging purposes while capturing retry output separately
                    retry_future = run_single_test(
                        module,
                        stdout=f'logs/test_{index}_{module}_retry.out',
                        stderr=f'logs/test_{index}_{module}_retry.err'
                    )
                    retry_future.result()  # Wait for retry completion
                    was_retried = True
                except Exception as e2:
                    # Check retry for exit code 5 as well
                    if 'unix exit code 5' in str(e2):
                        was_retried = True
                        # Treat as passed (skipped)
                    else:
                        results.append((module, f'FAILED after retry: {e2}', 0))
                        status = 'FAILED'
                        completed += 1
                        print(f"[{completed}/{len(futures)}] {module}: {status}")
                        continue  # Skip to next test - don't mark this as PASSED

        # Only reached if test passed (either on first attempt or after retry)
        # Read the output file to check for skipped tests
        # Use retry output file if test was retried
        if was_retried:
            output_file = f'logs/test_{index}_{module}_retry.out'
            error_file = f'logs/test_{index}_{module}_retry.err'
        else:
            output_file = f'logs/test_{index}_{module}.out'
            error_file = f'logs/test_{index}_{module}.err'
        
        skip_count = 0
        if os.path.exists(error_file):
            # Check the error file for NO TESTS RAN message (exit code 5)
            with open(error_file, 'r') as f:
                error_content = f.read()
                if 'NO TESTS RAN' in error_content and 'skipped=' in error_content:
                    match = re.search(r'NO TESTS RAN \(skipped=(\d+)\)', error_content)
                    if match:
                        skip_count = int(match.group(1))
        
        if skip_count == 0 and os.path.exists(output_file):
            with open(output_file, 'r') as f:
                output_content = f.read()
                match = re.search(r'OK \(skipped=(\d+)\)', output_content)
                if match:
                    skip_count = int(match.group(1))
                elif 'skipped' in output_content.lower():
                    skip_count = output_content.lower().count('skipped')

        results.append((module, 'PASSED', skip_count))
        status = f'PASSED (skipped={skip_count})' if skip_count > 0 else 'PASSED'
        
        completed += 1
        print(f"[{completed}/{len(futures)}] {module}: {status}")
        # Print output for successful tests and remove log files
        test_name = module
        test_case = ""
        mem_used = ""
        test_time = ""
        ok_status = "ok"
        # Example regex for extracting info
        match = re.search(
            r"(?P<test_case>[\w\.]+)\s+\.\.\.\s+Memory used:\s+(?P<mem>[\d\.]+)\s+GB\.\s+Test time:\s+(?P<time>[\d\.]+)\s+s", 
            output_content
        )
        if match:
            test_case = match.group("test_case")
            mem_used = match.group("mem")
            test_time = match.group("time")
            print(f"{test_name} ({test_case}) ...     Memory used: {mem_used} GB.  Test time: {test_time} s\n{ok_status}")
        # Note: output_content already printed above via regex match, no need to print again

        error_file = f'logs/test_{index}_{module}.err'
        try:
            os.remove(output_file)
            if os.path.exists(error_file):
                os.remove(error_file)
        except Exception:
            pass
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    return results, execution_time


def generate_summary_report(results, execution_time=0.0):
    """Generate a summary report of test execution in unittest style"""
    total_modules = len(results)
    passed_modules = sum(1 for _, status, _ in results if status == 'PASSED')
    failed_modules = total_modules - passed_modules
    total_skipped = sum(skip_count for _, status, skip_count in results if status == 'PASSED')
    
    # Create the unittest-style status line
    status_line = ""
    for module, status, skip_count in results:
        if status == 'PASSED':
            if skip_count > 0:
                status_line += "s" * skip_count + "."  # Add 's' for each skipped test
            else:
                status_line += "."
        else:
            status_line += "F"
    
    print(status_line)
    print("-" * 70)
    print(f"Ran {total_modules} test modules in {execution_time:.3f}s")
    print()
    
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
            if not status == 'PASSED':
                print(f"FAIL: {module}")
                print("-" * 70)
                print(f"Error: {status}")
                print()

def reload_parsl_config(max_workers=None, wait=1):
    """Safely reload Parsl with an updated config.

    - If a Parsl DFK exists, try to cleanup/shutdown it first.
    - Call `pl.clear()` to reset Parsl state.
    - Optionally set `config.max_workers` before loading.
    - Load the (possibly modified) config and wait briefly.
    """
    import os
    import signal
    import subprocess
    import time as _time

    def kill_interchange_processes(retries=3, delay=0.5):
        """Kill any running 'interchange.py' processes using multiple strategies."""
        for attempt in range(retries):
            killed_any = False
            # Try pkill first (simpler)
            try:
                subprocess.run(['pkill', '-f', 'interchange.py'], check=False)
            except Exception:
                pass

            # Try pgrep -> kill
            try:
                p = subprocess.run(['pgrep', '-f', 'interchange.py'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                if p.stdout:
                    for line in p.stdout.splitlines():
                        try:
                            pid = int(line.strip())
                            try:
                                os.kill(pid, signal.SIGTERM)
                                killed_any = True
                            except ProcessLookupError:
                                pass
                        except Exception:
                            pass
            except Exception:
                pass

            # Wait a bit for processes to die
            _time.sleep(delay)

            # Force kill remaining
            try:
                p = subprocess.run(['pgrep', '-f', 'interchange.py'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                if p.stdout:
                    for line in p.stdout.splitlines():
                        try:
                            pid = int(line.strip())
                            try:
                                os.kill(pid, signal.SIGKILL)
                                killed_any = True
                            except ProcessLookupError:
                                pass
                        except Exception:
                            pass
            except Exception:
                pass

            # If none found/killed, we're done
            p_check = subprocess.run(['pgrep', '-f', 'interchange.py'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            if not p_check.stdout:
                return True
            # otherwise retry
            _time.sleep(delay)
        # after retries, return False if still present
        final_check = subprocess.run(['pgrep', '-f', 'interchange.py'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        return not bool(final_check.stdout)

    # Attempt graceful DFK cleanup if present
    try:
        dfk = pl.dfk()  # may raise if no DFK
        if dfk is not None:
            try:
                dfk.cleanup()
            except Exception:
                pass
            try:
                # some Parsl versions expose shutdown
                getattr(dfk, "shutdown", lambda: None)()
            except Exception:
                pass
    except Exception:
        # no active DFK or older/newer API — continue
        pass

    # Clear Parsl global state
    try:
        pl.clear()
    except Exception:
        pass

    # Ensure any leftover interchange processes are killed before loading
    try:
        kill_interchange_processes()
    except Exception:
        pass

    # Apply requested changes
    if max_workers is not None:
        config.max_workers = max_workers

    # Attempt to load the (possibly updated) config, retrying if we detect leftover interchange processes
    last_exc = None
    for attempt in range(3):
        try:
            pl.load(config)
            break
        except AssertionError as ae:
            last_exc = ae
            # Often caused by existing interchange process; try killing and retry
            try:
                kill_interchange_processes()
            except Exception:
                pass
            _time.sleep(0.5)
        except Exception as e:
            last_exc = e
            # For other errors, try once more after a small delay
            _time.sleep(0.5)
    else:
        # Failed after retries; raise the last exception so caller can see details
        raise last_exc

    # Small pause to let executors start
    _time.sleep(wait)

    # Report status
    try:
        mw = getattr(config, "max_workers", None)
        print(f"Parsl loaded (max_workers={mw})")
        # show configured executors
        if hasattr(pl, "config") and pl.config is not None:
            try:
                exe_names = [type(e).__name__ for e in pl.config.executors]
                print("Executors:", exe_names)
            except Exception:
                pass
    except Exception:
        pass

    return pl
    """Safely reload Parsl with an updated config.

    - If a Parsl DFK exists, try to cleanup/shutdown it first.
    - Call `pl.clear()` to reset Parsl state.
    - Optionally set `config.max_workers` before loading.
    - Load the (possibly modified) config and wait briefly.
    """
    # Attempt graceful DFK cleanup if present
    try:
        dfk = pl.dfk()  # may raise if no DFK
        if dfk is not None:
            try:
                dfk.cleanup()
            except Exception:
                pass
    except Exception:
        # no active DFK or older/newer API — continue
        pass

    # Clear Parsl global state
    try:
        pl.clear()
    except Exception:
        pass

    # Apply requested changes
    if max_workers is not None:
        config.max_workers = max_workers

    # Load the (possibly updated) config
    pl.load(config)

    # Small pause to let executors start
    time.sleep(wait)

    # Report status
    try:
        mw = getattr(config, "max_workers", None)
        print(f"Parsl loaded (max_workers={mw})")
        # show configured executors
        if hasattr(pl, "config") and pl.config is not None:
            exe_names = [type(e).__name__ for e in pl.config.executors]
            print("Executors:", exe_names)
    except Exception:
        pass

    return pl


def main():
    """Main function to run parallel tests"""
    # Load Parsl configuration if not already loaded
    import parsl as pl
    try:
        # Check if Parsl is already configured
        pl.dfk()  # DataFlowKernel
        print("Parsl already configured.")
    except:
        # Parsl not configured, so load a configuration
        try:
            from parsl.configs.htex_local import config
            config.max_workers = 8  # processors
            pl.load(config)
            print("Parsl configuration loaded successfully.")
        except Exception as e:
            print(f"Error loading Parsl configuration: {e}")
            return
    
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
        except:
            pass
    
    # shutdown Parsl
    try:
        parsl.dfk().cleanup()
        parsl.dfk().shutdown()
    except Exception:
        pass

if __name__ == '__main__':
    main()