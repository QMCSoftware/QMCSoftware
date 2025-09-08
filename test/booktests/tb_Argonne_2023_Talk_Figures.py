import unittest
import subprocess
import os
import sys
import shutil
from testbook import testbook
from __init__ import TB_TIMEOUT, BaseNotebookTest

class NotebookTests(BaseNotebookTest):

    def setUp(self):
        super().setUp()
        # Install required packages
        subprocess.run(['pip', 'install', '-q', 'sympy', 'matplotlib', 'scipy'], check=False)
        # Create outputs directory if needed
        os.makedirs('outputs', exist_ok=True)
        # Copy pickle files from demos directory to current working directory
        demos_path = '../../demos/talk_paper_demos/Argonne_Talk_2023_May/'
        pickle_files = ['iid_ld.pkl', 'ld_parallel.pkl']
        for pkl_file in pickle_files:
            src = os.path.join(demos_path, pkl_file)
            if os.path.exists(src):
                shutil.copy2(src, pkl_file)
                print(f"\n\tCopied {src} to {pkl_file}")

    def tearDown(self):
        # move *eps to outputs
        for file in os.listdir('.'):
            if file.endswith('.eps') or file.endswith('.pkl'):
                shutil.move(file, os.path.join('outputs', file))
        super().tearDown()

    @testbook('../../demos/talk_paper_demos/Argonne_Talk_2023_May/Argonne_2023_Talk_Figures.ipynb', execute=False, timeout=TB_TIMEOUT)
    def test_argonne_talk_2023_figures_notebook(self, tb):
        # Execute cells up to but not including the stop_notebook cell
        for i in range(len(self.cells)):  
            if "qp.util.stop_notebook()" not in self.cells[i]['source']:
                self.execute_cell(i)
            else:
                break  # not running the rest of the notebook depending on umbridge and docker

if __name__ == '__main__':
    unittest.main()