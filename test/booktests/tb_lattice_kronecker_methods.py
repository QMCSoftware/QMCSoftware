import unittest
from __init__ import BaseNotebookTest

class NotebookTests(BaseNotebookTest):

    def test_lattice_kronecker_methods_notebook(self):
        self.run_notebook('../../demos/lattice_kronecker_methods.ipynb')

if __name__ == '__main__':
    unittest.main()
