import unittest, pytest
import subprocess
from __init__ import BaseNotebookTest

@pytest.mark.slow
class NotebookTests(BaseNotebookTest):

    def setUp(self):
        super().setUp()
        # Install required packages
        subprocess.run(['pip', 'install', '-q', 'matplotlib', 'scipy', 'scikit-learn', 'pandas'], check=False)

    def test_vectorized_qmc_notebook(self):
        notebook_path, _ = self.locate_notebook('../../demos/vectorized_qmc.ipynb')
        replacements = {
            'n = 2**6': 'n = 2**4',
            'tpax = 32': 'tpax = 8',
            "xplt = np.linspace(0,1,100)": "xplt = np.linspace(0,1,40)",
            'n_restarts_optimizer = 16': 'n_restarts_optimizer = 2',
            'abs_tol = 1e-3': 'abs_tol = 1e-1',
            'abs_tol=.025': 'abs_tol=.1',
            'abs_tol=.005': 'abs_tol=.05',
            'max_iter=1024': 'max_iter=200'
        }
        self.run_notebook(notebook_path, replacements)

if __name__ == '__main__':
    unittest.main()