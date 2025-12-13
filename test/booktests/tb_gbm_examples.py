import unittest
import os
import sys
from testbook import testbook
from __init__ import TB_TIMEOUT, BaseNotebookTest

class NotebookTests(BaseNotebookTest):

    def setUp(self):
        super().setUp()
        # Add the notebook's directory to sys.path for local imports
        self.notebook_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../demos/GBM/'))
        if self.notebook_dir not in sys.path:
            sys.path.insert(0, self.notebook_dir)
        # Change to notebook directory so relative imports work
        self.original_cwd = os.getcwd()
        os.chdir(self.notebook_dir)
    
    def tearDown(self):
        # Restore original working directory
        os.chdir(self.original_cwd)
        # Remove from sys.path
        if self.notebook_dir in sys.path:
            sys.path.remove(self.notebook_dir)
        super().tearDown()

    @testbook('../../demos/GBM/gbm_examples.ipynb', execute=True, timeout=TB_TIMEOUT)
    def test_examples_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()