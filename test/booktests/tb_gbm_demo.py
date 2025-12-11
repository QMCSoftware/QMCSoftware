import unittest
import subprocess
from testbook import testbook
from __init__ import TB_TIMEOUT, BaseNotebookTest

class NotebookTests(BaseNotebookTest):
    def setUp(self):
        super().setUp()  # Call parent setUp first to initialize timing attributes
        # Install required packages
        subprocess.run(['pip', 'install', '-q', 'scipy', 'matplotlib', 'ipywidgets', 'QuantLib==1.38'], check=False)

    @testbook('../../demos/GBM/gbm_demo.ipynb', execute=False, timeout=TB_TIMEOUT)
    def test_gbm_demo_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
