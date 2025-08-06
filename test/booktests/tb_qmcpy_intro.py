import unittest
from testbook import testbook
from __init__ import TB_TIMEOUT, BaseNotebookTest

class NotebookTests(BaseNotebookTest):

    @testbook('../../demos/qmcpy_intro.ipynb', execute=True, timeout=TB_TIMEOUT)
    def test_qmcpy_intro_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
