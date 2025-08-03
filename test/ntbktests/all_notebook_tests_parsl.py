#!/usr/bin/env python3
"""
Unit tests for demo notebooks execution using nbconvert and Parsl,
compatible with pytest discovery.
"""

import unittest
import os
import glob
import sys
from parsl import load, python_app
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
                max_blocks=1,
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
    print(f"{env['PYTHONPATH'] = }")
    # Use nbconvert to execute the notebook
    cmd = [
        sys.executable, '-m', 'nbconvert',
        '--to', 'notebook',
        '--execute',
        '--inplace',
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

class TestNotebooks(unittest.TestCase):
    """Dynamically generated notebook tests."""

# Dynamically attach test methods
all_nbs = glob.glob(os.path.join(DEMOS_PATH, '*.ipynb'))
for nb_file in all_nbs:
    nb_name = os.path.basename(nb_file)
    if nb_name in exclude_notebooks:
        continue
    rel_path = os.path.join(DEMOS_PATH, nb_name)
    test_name = f'test_{nb_name.replace(".ipynb","").replace("-","_")}_notebook'

    def make_test(path):
        def test(self):
            future = run_notebook(path)
            result = future.result()
            self.assertEqual(result, path)
        return test

    setattr(TestNotebooks, test_name, make_test(rel_path))

if __name__ == '__main__':
    import unittest
    unittest.main()
