import unittest
from testbook import testbook
from __init__ import TB_TIMEOUT, BaseNotebookTest

class NotebookTests(BaseNotebookTest):

    @unittest.skip("Run time error")
    @testbook('../../demos/gaussian_diagnostics/gaussian_diagnostics_demo.ipynb', execute=True, timeout=TB_TIMEOUT)
    def test_gaussian_diagnostics_demo_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()