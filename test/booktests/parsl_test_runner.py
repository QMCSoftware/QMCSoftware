import parsl
from parsl import bash_app
import glob
import os
import re
import time
import platform
from pathlib import Path
import time
import parsl as pl

# Use ThreadPoolExecutor on macOS (avoids interchange.py spawn issue)
# Use HighThroughputExecutor on Linux for better performance
if platform.system() == 'Darwin':
    from parsl.config import Config
    from parsl.executors import ThreadPoolExecutor
    config = Config(executors=[ThreadPoolExecutor(max_threads=8, label="local_threads")])
else:
    from parsl.configs.htex_local import config



@bash_app
def run_single_test(test_file, stdout='test_output.txt', stderr='test_error.txt'):
    """Run a single test file using bash"""
    return f"""
    PYTHONWARNINGS="ignore::UserWarning,ignore::DeprecationWarning,ignore::FutureWarning,ignore::ImportWarning" python -m coverage run --append --source=../../qmcpy/ -m unittest {test_file} 2>&1
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
    total_test_time = 0.0  # Track sum of individual test execution times
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
                # Read and print error log contents for debugging
                stdout_file = f'logs/test_{index}_{module}.out'
                print(f"\n--- Error details for {module} (first attempt) ---")
                if os.path.exists(stdout_file):
                    try:
                        with open(stdout_file, 'r') as f:
                            content = f.read()
                            # Print last 2000 chars to avoid log overflow
                            print(content[-2000:] if len(content) > 2000 else content)
                    except Exception as read_err:
                        print(f"Could not read log file: {read_err}")
                print(f"--- End of error details for {module} ---\n")
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
                        # Read and print retry error log contents for debugging
                        retry_stdout_file = f'logs/test_{index}_{module}_retry.out'
                        print(f"\n--- Failure details for {module} (after retry) ---")
                        if os.path.exists(retry_stdout_file):
                            try:
                                with open(retry_stdout_file, 'r') as f:
                                    content = f.read()
                                    # Print last 2000 chars to avoid log overflow
                                    print(content[-2000:] if len(content) > 2000 else content)
                            except Exception as read_err:
                                print(f"Could not read log file: {read_err}")
                        print(f"--- End of failure details for {module} ---\n")
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
        output_content = ""
        
        # Read output and error files once
        if os.path.exists(error_file):
            # Check the error file for NO TESTS RAN message (exit code 5)
            with open(error_file, 'r') as f:
                error_content = f.read()
                if 'NO TESTS RAN' in error_content and 'skipped=' in error_content:
                    match = re.search(r'NO TESTS RAN \(skipped=(\d+)\)', error_content)
                    if match:
                        skip_count = int(match.group(1))
        
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                output_content = f.read()
                if skip_count == 0:
                    match = re.search(r'OK \(skipped=(\d+)\)', output_content)
                    if match:
                        skip_count = int(match.group(1))
                    elif 'skipped' in output_content.lower():
                        skip_count = output_content.lower().count('skipped')

        results.append((module, 'PASSED', skip_count))
        status = f'PASSED (skipped={skip_count})' if skip_count > 0 else 'PASSED'
        
        completed += 1
        
        # Read error file content for test details
        error_content = ""
        if os.path.exists(error_file):
            with open(error_file, 'r') as f:
                error_content = f.read()
        
        # Extract memory and time from stdout, test case name from module
        test_name = module
        ok_status = "ok"
        mem_match = re.search(r"Memory used:\s*(?P<mem>[\d\.]+)\s*GB\.\s*Test time:\s*(?P<time>[\d\.]+)\s*s", output_content)
        if mem_match:
            mem_used = mem_match.group("mem")
            test_time = mem_match.group("time")
            total_test_time += float(test_time)  # Accumulate individual test times
            print(f"[{completed}/{len(futures)}] {module}: {status}")
            print(f"{test_name} ...     Memory used: {mem_used} GB.  Test time: {test_time} s\n{ok_status}")
        else:
            print(f"[{completed}/{len(futures)}] {module}: {status}")
        
        # Note: Keep log files for debugging (don't delete them)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    return results, execution_time, total_test_time


def generate_summary_report(results, execution_time=0.0, total_test_time=0.0):
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
    if total_test_time > 0:
        print(f"Total test time: {total_test_time:.3f}s (overhead: {execution_time - total_test_time:.3f}s)")
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



def main():
    """Main function to run parallel tests"""
    # Load Parsl configuration if not already loaded
    import parsl as pl
    import platform
    import os
    import sys
    try:
        # Check if Parsl is already configured
        pl.dfk()  # DataFlowKernel
        print("Parsl already configured.")
    except:
        # Parsl not configured, so load a configuration
        try:
            # Read max workers from environment if provided
            try:
                max_workers = int(os.environ.get('PARSL_MAX_WORKERS'))
            except Exception:
                max_workers = 8  # default
            
            if platform.system() == 'Darwin':
                from parsl.config import Config
                from parsl.executors import ThreadPoolExecutor
                config = Config(executors=[ThreadPoolExecutor(max_threads=max_workers, label="local_threads")])
            else:
                from parsl.configs.htex_local import config
                config.max_workers = max_workers
            pl.load(config)
            print("Parsl configuration loaded successfully.")
        except Exception as e:
            print(f"Error loading Parsl configuration: {e}")
            sys.exit(1)
    
    results = None
    try:
        results, execution_time, total_test_time = execute_parallel_tests()
        generate_summary_report(results, execution_time, total_test_time)
    except Exception as e:
        print(f"Error during parallel execution: {e}")
        sys.exit(1)
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
    
    # Exit with non-zero code if any tests failed
    if results:
        failed_modules = sum(1 for _, status, _ in results if status != 'PASSED')
        if failed_modules > 0:
            sys.exit(1)

if __name__ == '__main__':
    main()