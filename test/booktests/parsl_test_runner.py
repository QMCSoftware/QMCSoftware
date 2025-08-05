import parsl
from parsl import bash_app
import glob
import os
from pathlib import Path

@bash_app
def run_single_test(test_file, stdout='test_output.txt', stderr='test_error.txt'):
    """Run a single test file using bash"""
    return f"""
    PYTHONWARNINGS="ignore::UserWarning,ignore::DeprecationWarning,ignore::FutureWarning,ignore::ImportWarning" python -m unittest {test_file}
    """

def execute_parallel_tests():
    """Execute all testbook tests in parallel using Parsl"""
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
        futures.append((module, future))
    
    print("All tests submitted to Parsl executor...")
    
    # Wait for completion and collect results
    results = []
    completed = 0
    for module, future in futures:
        try:
            future.result()  # Wait for completion
            results.append((module, 'PASSED'))
            status = 'PASSED'
        except Exception as e:
            results.append((module, f'FAILED: {e}'))
            status = 'FAILED'
        
        completed += 1
        print(f"[{completed}/{len(futures)}] {module}: {status}")
    
    return results

def generate_summary_report(results):
    """Generate a summary report of test execution"""
    total = len(results)
    passed = sum(1 for _, status in results if status == 'PASSED')
    failed = total - passed
    
    print(f"PARALLEL TEST EXECUTION SUMMARY")
    print(f"{'='*50}")
    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"\n{'='*50}")
    if total > 0:
        print(f"Success rate: {(passed/total)*100:.1f}%")
    else:
        print("Success rate: N/A (no tests found)")
    
    if failed > 0:
        print(f"\nFailed tests:")
        for module, status in results:
            if status != 'PASSED':
                print(f"  - {module}: {status}")

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
            pl.load(config)
            print("Parsl configuration loaded successfully.")
        except Exception as e:
            print(f"Error loading Parsl configuration: {e}")
            return
    
    try:
        results = execute_parallel_tests()
        generate_summary_report(results)
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