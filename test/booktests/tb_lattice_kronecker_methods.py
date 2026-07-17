import unittest
from testbook import testbook
from __init__ import TB_TIMEOUT, BaseNotebookTest

class NotebookTests(BaseNotebookTest):

    @testbook('../../demos/lattice_kronecker_methods.ipynb', execute=True, timeout=TB_TIMEOUT)
    def test_lattice_kronecker_methods_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
