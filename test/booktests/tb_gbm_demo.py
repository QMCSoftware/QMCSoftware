import unittest
from testbook import testbook
from __init__ import TB_TIMEOUT, BaseNotebookTest

class NotebookTests(BaseNotebookTest):
    def setUp(self):
        super().setUp()  # Call parent setUp first to initialize timing attributes

    @unittest.skip("Skipping GBM demo notebook test for now.")
    @testbook('../../demos/GBM/gbm_demo.ipynb', execute=True, timeout=TB_TIMEOUT)
    def test_gbm_demo_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
