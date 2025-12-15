import os
import sys
import unittest
from testbook import testbook
from __init__ import TB_TIMEOUT, BaseNotebookTest
class NotebookTests(BaseNotebookTest):
    
    def test_gbm_examples_notebook(self):
        notebook_path, notebook_dir = self.locate_notebook('../../demos/GBM/gbm_examples.ipynb')
        self.fix_gbm_symlinks(notebook_dir)
        self.run_notebook(notebook_path, notebook_dir)

if __name__ == '__main__':
    unittest.main()