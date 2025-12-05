import unittest
import os
import subprocess
from testbook import testbook
import sys
sys.path.insert(0, os.path.dirname(__file__))
from __init__ import TB_TIMEOUT, BaseNotebookTest

@unittest.skip("Skipping NotebookTests class")
class NotebookTests(BaseNotebookTest):
    def setUp(self):
        super().setUp()  # Call parent setUp first to initialize timing attributes
        subprocess.run(['pip', 'install', '-q', 'tueplots'], check=False)
        os.makedirs('outputs', exist_ok=True)
    
    @testbook('../../demos/talk_paper_demos/ACMTOMS_Sorokin_2025/acm_toms_sorokin_2025.ipynb', execute=False, timeout=TB_TIMEOUT)
    def test_acm_toms_sorokin_2025_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
