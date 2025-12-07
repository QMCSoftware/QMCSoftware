import unittest
import subprocess
from testbook import testbook
from __init__ import TB_TIMEOUT, BaseNotebookTest

class NotebookTests(BaseNotebookTest):

    def setUp(self):
        super().setUp()
        # Install required packages
        subprocess.run(['pip', 'install', '-q', 'matplotlib', 'scipy', 'scikit-learn', 'pandas'], check=False)

    @testbook('../../demos/vectorized_qmc.ipynb', execute=True, timeout=TB_TIMEOUT)
    def test_vectorized_qmc_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()