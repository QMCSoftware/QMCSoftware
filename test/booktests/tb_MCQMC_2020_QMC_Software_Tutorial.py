import unittest
import subprocess
from testbook import testbook
from __init__ import TB_TIMEOUT, BaseNotebookTest

@unittest.skip("Skipping notebook tests for now")
class NotebookTests(BaseNotebookTest):

    def setUp(self):
        # Call parent setUp first to initialize timing attributes
        super().setUp()
        subprocess.run(['pip', 'install', '-q', 'matplotlib', 'scipy', 'numpy'], check=False)   
    
    @testbook('../../demos/talk_paper_demos/MCQMC_Tutorial_2020/MCQMC_2020_QMC_Software_Tutorial.ipynb', execute=True, timeout=TB_TIMEOUT)
    def test_mcqmc_2020_qmc_software_tutorial_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
