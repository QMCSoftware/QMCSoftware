import parsl
from parsl import bash_app
import glob
import os
import re
import time
from pathlib import Path

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
    
    # Get all test files - need to find them relative to the current QMCSoftware directory 
    test_files = glob.glob('tb_*.py')
    test_modules = [os.path.basename(f).replace('.py', '') for f in test_files]
    
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
    for module, future, index in futures:
        try:
            future.result()  # Wait for completion
            
            # Read the output file to check for skipped tests
            output_file = f'logs/test_{index}_{module}.out'
            skip_count = 0
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    output_content = f.read()
                    # Check for unittest skip messages
                    import re
                    match = re.search(r'OK \(skipped=(\d+)\)', output_content)
                    if match:
                        skip_count = int(match.group(1))
                    elif 'skipped' in output_content.lower():
                        # Count occurrences of "skipped" in the output as fallback
                        skip_count = output_content.lower().count('skipped')
            
            results.append((module, 'PASSED', skip_count))
            status = f'PASSED (skipped={skip_count})' if skip_count > 0 else 'PASSED'
        except Exception as e:
            results.append((module, f'FAILED: {e}', 0))
            status = 'FAILED'
        
        completed += 1
        print(f"[{completed}/{len(futures)}] {module}: {status}")
    
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


def main():
    """Main function to run parallel tests"""
    # Load Parsl configuration if not already loaded
    import parsl as pl
    try:
        # Check if Parsl is already configured
        pl.dfk()  # DataFlowKernel
        print("Parsl already configured.")
    except pl.errors.NoDataFlowKernelError:
        # Parsl not configured, so load a configuration
        try:
            from parsl.configs.htex_local import config
            config.executors[0].max_workers = 4  # Set to 4 workers to reduce memory usage
            pl.load(config)
            print("Parsl configuration loaded successfully.")
        except Exception as e:
            print(f"Error loading Parsl configuration: {e}")
            return None
    except Exception as e:
        print(f"Unexpected error checking Parsl configuration: {e}")
        return None
    
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