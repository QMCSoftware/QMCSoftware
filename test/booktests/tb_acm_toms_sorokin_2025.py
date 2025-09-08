import unittest
import os
import subprocess
from testbook import testbook
from __init__ import TB_TIMEOUT, BaseNotebookTest

class NotebookTests(BaseNotebookTest):
    def setUp(self):
        # Call parent setUp first to initialize timing attributes
        super().setUp()
        subprocess.run(['pip', 'install', '-q', 'tueplots'], check=False)
        os.makedirs('demos/talk_paper_demos/ACMTOMS_Sorokin_2025/outputs', exist_ok=True)
        os.makedirs('outputs', exist_ok=True)
    
    @testbook('../../demos/talk_paper_demos/ACMTOMS_Sorokin_2025/acm_toms_sorokin_2025.ipynb', execute=True, timeout=TB_TIMEOUT)
    def test_acm_toms_sorokin_2025_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
