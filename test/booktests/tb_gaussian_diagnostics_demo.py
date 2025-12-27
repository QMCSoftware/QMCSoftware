import unittest
import subprocess
import os
from testbook import testbook
from __init__ import TB_TIMEOUT, BaseNotebookTest

@pytest.mark.slow 
class NotebookTests(BaseNotebookTest):

    def setUp(self):
        super().setUp()
        # Install required packages
        subprocess.run(['pip', 'install', '-q', 'matplotlib', 'scipy', 'seaborn==0.8'], check=False)
        # Create outputs directory if needed
        os.makedirs('outputs_nb', exist_ok=True)

    @testbook('../../demos/gaussian_diagnostics/gaussian_diagnostics_demo.ipynb', execute=True, timeout=TB_TIMEOUT)
    def test_gaussian_diagnostics_demo_notebook(self, tb):
        # Execute cells up to but not including the stop_notebook cell
        for i in range(len(self.cells)):  
            if "plt.style.use('seaborn-v0_8-poster')" not in self.cells[i]['source']:
                self.execute_cell(i)
            else:
                break  # not running the rest of the notebook depending on umbridge and docker

if __name__ == '__main__':
    unittest.main()