#!/usr/bin/env python3
"""
Parallel notebook tests using Parsl and nbconvert, located in QMCSoftware/test/ntbktests.
This script executes demo notebooks in parallel via subprocess calls to nbconvert.
"""

import os
import glob
import sys
import subprocess
from parsl import load, python_app
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider

# Ensure project modules are found
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ['PYTHONPATH'] = project_root + os.pathsep + os.environ.get('PYTHONPATH', '')

# Parsl configuration with enhanced environment setup
config = Config(
    executors=[
        HighThroughputExecutor(
            label="htex_local",
            cores_per_worker=1,
            worker_debug=True,  # Enable debug to see worker issues
            provider=LocalProvider(
                init_blocks=1, 
                max_blocks=1,
                worker_init=f'''
export PYTHONPATH={project_root}:$PYTHONPATH
cd {project_root}
echo "Worker starting in directory: $(pwd)"
echo "Worker PYTHONPATH: $PYTHONPATH"
python -c "import sys; print('Worker sys.path:', sys.path)"
python -c "
try:
    import qmcpy
    print('qmcpy import test successful')
except ImportError as e:
    print('qmcpy import failed:', e)
    import sys
    print('Available paths:', sys.path)
"
'''
            ),
        )
    ]
)
dfk = load(config)

# Demos directory
DEMOS_PATH = os.path.join(project_root, 'demos')

# Notebooks to skip
exclude = {
    'linear-scrambled-halton.ipynb', 'sample_scatter_plots.ipynb', 'iris.ipynb',
    'digital_net_b2.ipynb', 'MC_vs_QMC.ipynb', 'quasirandom_generators.ipynb',
    'asian-option-mlqmc.ipynb', 'gaussian_diagnostics_demo.ipynb',
    'importance_sampling.ipynb', 'lattice_random_generator.ipynb',
    'ld_randomizations_and_higher_order_nets.ipynb', 'PricingAsianOptions.ipynb',
    'vectorized_qmc_bayes.ipynb', 'prob_failure_gp_ci.ipynb', 'umbridge.ipynb',
    'dakota_genz.ipynb', 'vectorized_qmc.ipynb', 'pydata.chi.2023.ipynb'
}

@python_app
def run_notebook(path, project_root):
    """
    Execute a notebook via nbconvert in a subprocess.
    Returns the notebook path if successful.
    Raises RuntimeError on failure.
    """
    import os
    import sys
    import subprocess
    
    # Create environment with proper PYTHONPATH
    env = os.environ.copy()
    env['PYTHONPATH'] = project_root + os.pathsep + env.get('PYTHONPATH', '')
    
    # Debug: Print the environment
    print(f"Running notebook: {os.path.basename(path)}")
    print(f"Project root: {project_root}")
    print(f"PYTHONPATH: {env.get('PYTHONPATH', 'NOT SET')}")
    
    nb_dir = os.path.dirname(path)
    cmd = [
        sys.executable, '-m', 'nbconvert',
        '--to', 'notebook', '--execute', path,
        '--ExecutePreprocessor.timeout=600'
    ]
    proc = subprocess.run(cmd, cwd=nb_dir, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        print(f"Notebook {os.path.basename(path)} failed:")
        print(f"STDOUT: {proc.stdout.decode()}")
        print(f"STDERR: {proc.stderr.decode()}")
        raise RuntimeError(f"Notebook failed: {proc.stderr.decode()}")
    print(f"Notebook {os.path.basename(path)} completed successfully")
    return path

def main():
    notebooks = glob.glob(os.path.join(DEMOS_PATH, '*.ipynb'))
    tasks = []
    print("Dispatching tests...")
    for nb in notebooks:
        name = os.path.basename(nb)
        if name in exclude:
            print("Skipping", name)
            continue
        tasks.append(run_notebook(nb, project_root))

    results = [t.result() for t in tasks]
    print("Completed:", len(results), "notebooks")
    for r in results:
        print(" -", r)

if __name__ == '__main__':
    main()
