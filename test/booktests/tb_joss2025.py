import unittest
import os
import shutil
import subprocess
from __init__ import TB_TIMEOUT, BaseNotebookTest

class NotebookTests(BaseNotebookTest):
    def setUp(self):
        super().setUp()  # Call parent setUp first to initialize timing attributes
        subprocess.run([
            'pip', 'install', '-q', 
            'seaborn>=0.13.0', 
            'tueplots'
        ], check=False)
        # Create the JOSS2025.outputs directory that the notebook expects
        self.output_dir = os.path.join(os.path.dirname(__file__), 'JOSS2025.outputs')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
            self._created_output_dir = True
        else:
            self._created_output_dir = False
     
        
    def tearDown(self):
        # Clean up the created directory if we created it   
        if hasattr(self, '_created_output_dir') and self._created_output_dir:
            if os.path.exists(self.output_dir):
                shutil.rmtree(self.output_dir)
        super().tearDown()


    def test_joss2025_notebook(self):
        # locate_notebook will raise unittest.SkipTest if notebook not present
        notebook_path, notebook_dir = self.locate_notebook('../../demos/talk_paper_demos/JOSS2025/joss2025.ipynb')
        self.change_notebook_cells(notebook_path, 
                                   replacements={"trials = 100": "trials = 2",
                                                 "assert os.path.isdir(OUTDIR)":"",
                                                 'fig.savefig':'#fig.savefig',
                                                 'np.save':'#np.save',
                                                 })        
        self.run_notebook(notebook_path, timeout=TB_TIMEOUT)

if __name__ == '__main__':
    unittest.main()