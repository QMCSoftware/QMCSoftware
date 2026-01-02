import unittest, pytest
from __init__ import BaseNotebookTest

class NotebookTests(BaseNotebookTest):

    def test_vectorized_qmc_bayes_notebook(self):
        notebook_path, _ = self.locate_notebook('../../demos/vectorized_qmc_bayes.ipynb')
        # Reduce sample/grid sizes and loosen tolerances to speed execution
        replacements = {
            'n = 2**6': 'n = 2**4',
            'tpax = 32': 'tpax = 8',
            'abs_tol = 1e-3': 'abs_tol = 1e-1',
            'abs_tol=.025': 'abs_tol=.1',
            'abs_tol=.05': 'abs_tol=.1',
            'abs_tol=.005': 'abs_tol=.05',
            'n_restarts_optimizer = 16': 'n_restarts_optimizer = 2',
            'n_limit=2**18': 'n_limit=2**12',
            'xplt = np.linspace(0,1,100)': 'xplt = np.linspace(0,1,40)'
        }
        self.run_notebook(notebook_path, replacements)

if __name__ == '__main__':
    unittest.main()
