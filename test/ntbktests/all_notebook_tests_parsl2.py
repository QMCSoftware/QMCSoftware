#!/usr/bin/env python3
"""
Custom parallel notebook test runner using Parsl with individual reporting.
Bypasses unittest for true parallel execution with detailed per-notebook results.
"""

import os
import glob
import sys
import time
from parsl import load, python_app, clear
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider

# Ensure the project root is on the Python path for notebook imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Export PYTHONPATH for Parsl worker processes
os.environ['PYTHONPATH'] = project_root + os.pathsep + os.environ.get('PYTHONPATH', '')

# Configure Parsl HighThroughputExecutor with proper environment setup
config = Config(
    executors=[
        HighThroughputExecutor(
            label="htex_local",
            cores_per_worker=1,
            worker_debug=True,
            provider=LocalProvider(
                init_blocks=1, 
                max_blocks=5,  # Allow up to 5 parallel notebook executions
                worker_init=f'''
export PYTHONPATH={project_root}:$PYTHONPATH
cd {project_root}
echo "Worker PYTHONPATH: $PYTHONPATH"
python -c "
import sys
sys.path.insert(0, '{project_root}')
try:
    import qmcpy
    print('qmcpy import successful in worker')
except Exception as e:
    print('qmcpy import failed in worker:', e)
"
'''
            ),
        )
    ]
)
dfk = load(config)

# Path to the demos directory
DEMOS_PATH = os.path.join(project_root, 'demos')

# Notebooks to skip
exclude_notebooks = {
    'linear-scrambled-halton.ipynb',
    'sample_scatter_plots.ipynb',
    'iris.ipynb',
    'digital_net_b2.ipynb',
    'MC_vs_QMC.ipynb',
    'quasirandom_generators.ipynb',
    'asian-option-mlqmc.ipynb',
    'gaussian_diagnostics_demo.ipynb',
    'importance_sampling.ipynb',
    'lattice_random_generator.ipynb',
    'ld_randomizations_and_higher_order_nets.ipynb',
    'PricingAsianOptions.ipynb',
    'vectorized_qmc_bayes.ipynb',
    'prob_failure_gp_ci.ipynb',
    'umbridge.ipynb',
    'dakota_genz.ipynb',
    'vectorized_qmc.ipynb',
    'pydata.chi.2023.ipynb',
}

@python_app
def run_notebook(path):
    """
    Parsl task to execute a notebook via nbconvert subprocess.
    Returns the notebook path on success.
    """
    import os
    import sys
    import subprocess
    
    # Hardcode the project root path to avoid __file__ serialization issues
    project_root = '/Users/terrya/Documents/ProgramData/QMCSoftware'
    
    nb_path = os.path.abspath(path)
    nb_dir = os.path.dirname(nb_path)

    # Create environment with proper PYTHONPATH
    env = os.environ.copy()
    env['PYTHONPATH'] = project_root + os.pathsep + env.get('PYTHONPATH', '')
    
    # Use nbconvert to execute the notebook
    cmd = [
        sys.executable, '-m', 'nbconvert',
        '--to', 'notebook',
        '--execute',
        '--stdout',
        nb_path,
        '--ExecutePreprocessor.timeout=600'
    ]
    
    try:
        result = subprocess.run(
            cmd, 
            cwd=nb_dir, 
            env=env,
            capture_output=True,
            text=True,
            check=True
        )
        return path
    except subprocess.CalledProcessError as e:
        error_msg = f"Notebook execution failed:\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}"
        raise RuntimeError(error_msg)

def cleanup_parsl_workers():
    """Clean up any remaining Parsl worker processes"""
    import psutil
    
    current_pid = os.getpid()
    terminated_count = 0
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if (proc.info['pid'] != current_pid and 
                (proc.info['name'] == 'Python' or proc.info['name'] == 'python')):
                cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                
                # Look for Parsl-specific patterns
                if any(pattern in cmdline for pattern in [
                    'parsl.executors', 'interchange.py', 'process_worker_pool.py',
                    'htex', 'worker.py'
                ]):
                    print(f"Terminating Parsl worker process: {proc.info['pid']}")
                    proc.terminate()
                    terminated_count += 1
                    
                    # Wait a bit, then force kill if still running
                    try:
                        proc.wait(timeout=2)
                    except psutil.TimeoutExpired:
                        proc.kill()
                        print(f"Force killed stubborn process: {proc.info['pid']}")
                        
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    
    if terminated_count > 0:
        print(f"Cleaned up {terminated_count} Parsl worker processes")

if __name__ == '__main__':
    """Custom parallel test runner with individual notebook reporting"""
    
    # Get all notebook paths
    all_nbs = glob.glob(os.path.join(DEMOS_PATH, '*.ipynb'))
    notebook_paths = [nb for nb in all_nbs if os.path.basename(nb) not in exclude_notebooks]
    
    print("="*60)
    print(f"PARALLEL NOTEBOOK EXECUTION")
    print("="*60)
    print(f"Running {len(notebook_paths)} notebooks in parallel...")
    print(f"Max parallel workers: 5")
    print("-"*60)
    
    start_time = time.time()
    
    # Submit all notebooks as Parsl futures
    futures = [run_notebook(path) for path in notebook_paths]
    
    # Collect results with individual reporting
    passed = []
    failed = []
    
    for future, path in zip(futures, notebook_paths):
        notebook_name = os.path.basename(path)
        try:
            result = future.result()
            print(f"✓ PASS: {notebook_name}")
            passed.append(notebook_name)
        except Exception as e:
            print(f"✗ FAIL: {notebook_name}")
            print(f"  Error: {str(e)[:100]}...")  # Truncate long error messages
            failed.append((notebook_name, str(e)))
    
    end_time = time.time()
    
    # Final summary report
    print("="*60)
    print("EXECUTION SUMMARY")
    print("="*60)
    print(f"Total notebooks: {len(notebook_paths)}")
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    print(f"PASSED: {len(passed)}")
    print(f"FAILED: {len(failed)}")
    print("-"*60)
    
    if passed:
        print("PASSED NOTEBOOKS:")
        for nb in passed:
            print(f"  ✓ {nb}")
        print()
    
    if failed:
        print("FAILED NOTEBOOKS:")
        for nb, error in failed:
            print(f"  ✗ {nb}")
            print(f"    {error[:200]}...")  # Show first 200 chars of error
        print()
    
    # Clean up Parsl
    print("Cleaning up Parsl...")
    clear()
    cleanup_parsl_workers()
    
    print("="*60)
    if failed:
        print(f"RESULT: FAILED ({len(failed)} notebooks failed)")
        sys.exit(1)
    else:
        print("RESULT: ALL TESTS PASSED")
        sys.exit(0)

# run this script with:
# time python all_notebook_tests_parsl2.py
# python all_notebook_tests_parsl2.py  0.40s user 0.48s system 0% cpu 2:03.10 total