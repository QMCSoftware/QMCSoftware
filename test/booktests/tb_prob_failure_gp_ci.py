import unittest
import os
import shutil
from testbook import testbook
from __init__ import TB_TIMEOUT, BaseNotebookTest

class NotebookTests(BaseNotebookTest):

    def setUp(self):
        super().setUp()
        # Create outputs directory
        os.makedirs('outputs', exist_ok=True)
    
    def tearDown(self):
        # move *eps to outputs
        for file in os.listdir('.'):
            if file.endswith('.eps') or file.endswith('.pkl'):
                shutil.move(file, os.path.join('outputs', file))
        super().tearDown()

    @testbook('../../demos/talk_paper_demos/ProbFailureSorokinRao/prob_failure_gp_ci.ipynb', execute=True, timeout=TB_TIMEOUT)
    def test_prob_failure_gp_ci_notebook(self, tb):
        # Execute cells up to but not including the stop_notebook cell
        for i in range(len(self.cells)):  
            if "import umbridge" not in self.cells[i]['source']:
                self.execute_cell(i)
            else:
                break  # not running the rest of the notebook depending on umbridge and docker

if __name__ == '__main__':
    unittest.main()
