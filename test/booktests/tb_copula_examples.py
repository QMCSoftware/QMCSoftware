import unittest
from testbook import testbook
from __init__ import TB_TIMEOUT, BaseNotebookTest

class NotebookTests(BaseNotebookTest):

    @testbook('../../demos/copula_examples.ipynb', execute=True, timeout=TB_TIMEOUT)
    def test_copula_examples_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
