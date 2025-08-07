import unittest
from testbook import testbook
from __init__ import TB_TIMEOUT, BaseNotebookTest

class NotebookTests(BaseNotebookTest):

    @unittest.skip("Runtime error")
    @testbook('../../demos/some_true_measures.ipynb', execute=True,timeout=TB_TIMEOUT)
    def test_some_true_measures_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
