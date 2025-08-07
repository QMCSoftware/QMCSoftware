import unittest
from testbook import testbook
from __init__ import TB_TIMEOUT, BaseNotebookTest

class NotebookTests(BaseNotebookTest):

    @unittest.skip("Runtime error")
    @testbook('../../demos/pydata.chi.2023.ipynb', execute=True, timeout=TB_TIMEOUT)
    def test_pydata_chi_2023_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
