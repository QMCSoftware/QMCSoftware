import unittest
from testbook import testbook
from __init__ import TB_TIMEOUT, BaseNotebookTest

class NotebookTests(BaseNotebookTest):

    @unittest.skip("Runtime error")
    @testbook('../../demos/vectorized_qmc.ipynb', execute=True, timeout=TB_TIMEOUT)
    def test_vectorized_qmc_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()