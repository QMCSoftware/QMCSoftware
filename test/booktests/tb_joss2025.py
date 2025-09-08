import unittest
import os
import shutil
import subprocess
from testbook import testbook
from __init__ import TB_TIMEOUT, BaseNotebookTest

class NotebookTests(BaseNotebookTest):
    def setUp(self):
        super().setUp()  # Call parent setUp first to initialize timing attributes
        # Install compatible package versions
        subprocess.run(['pip', 'install', '-q', 'seaborn', 'tueplots'], check=False)
        # Create the JOSS2025.outputs directory that the notebook expects
        self.output_dir = os.path.join(os.path.dirname(__file__), 'JOSS2025.outputs')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        
    def tearDown(self):
        # Clean up the created directory if we created it
        if hasattr(self, '_created_output_dir') and self._created_output_dir:
            if os.path.exists(self.output_dir):
                shutil.rmtree(self.output_dir)
        super().tearDown()

    @testbook('../../demos/talk_paper_demos/JOSS2025/joss2025.ipynb', execute=True, timeout=TB_TIMEOUT)
    def test_joss2025_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
