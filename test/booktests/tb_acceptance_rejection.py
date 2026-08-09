import unittest
from testbook import testbook
from __init__ import TB_TIMEOUT, BaseNotebookTest

class NotebookTests(BaseNotebookTest):

    @testbook('../../demos/acceptance_rejection.ipynb', execute=True, timeout=TB_TIMEOUT)
    def test_acceptance_rejection_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
