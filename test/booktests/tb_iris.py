import unittest, pytest
from __init__ import BaseNotebookTest


class NotebookTests(BaseNotebookTest):

    def test_iris_notebook(self):
        notebook_path, _ = self.locate_notebook('../../demos/iris.ipynb')
        # Speed up notebook execution by loosening tolerances and reducing search sizes
        replacements = {
            'abs_tol=1e-4': 'abs_tol=1e-1',
            'abs_tol=2.5e-2': 'abs_tol=1e-1',
            'abs_tol=5e-2': 'abs_tol=1e-1',
            'abs_tol=1e-3': 'abs_tol=1e-1',
            'n_calls = 32': 'n_calls = 6',
            'nticks = 32': 'nticks = 8',
            'n_init=2**10': 'n_init=2**6'
        }
        self.run_notebook(notebook_path, replacements)

if __name__ == '__main__':
    unittest.main()
