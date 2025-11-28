import unittest
from testbook import testbook
from __init__ import TB_TIMEOUT, BaseNotebookTest

class NotebookTests(BaseNotebookTest):

    def setUp(self):
        super().setUp()

    @testbook('../../demos/umbridge.ipynb', execute=False, timeout=TB_TIMEOUT)
    def test_umbridge_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
