import unittest
from testbook import testbook
from __init__ import TB_TIMEOUT, BaseNotebookTest

class NotebookTests(BaseNotebookTest):

    def setUp(self):
        super().setUp()

    @unittest.skip("Requires external server - umbridge")
    @testbook('../../demos/umbridge.ipynb', execute=True, timeout=TB_TIMEOUT)
    def test_umbridge_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
