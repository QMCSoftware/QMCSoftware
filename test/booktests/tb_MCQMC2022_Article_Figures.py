import unittest
import subprocess
import os
from testbook import testbook
from __init__ import TB_TIMEOUT, BaseNotebookTest

class NotebookTests(BaseNotebookTest):

    def setUp(self):
        super().setUp()
        # Install required packages
        subprocess.run(['pip', 'install', '-q', 'matplotlib', 'umbridge'], check=False)
        # Create outputs directory if needed
        os.makedirs('outputs', exist_ok=True)

    @unittest.skip("Requires external server - umbridge")
    @testbook('../../demos/talk_paper_demos/MCQMC2022_Article_Figures/MCQMC2022_Article_Figures.ipynb', execute=True, timeout=TB_TIMEOUT)
    def test_mcqmc2022_article_figures_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
