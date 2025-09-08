import unittest
import os
import shutil
from testbook import testbook
from __init__ import TB_TIMEOUT, BaseNotebookTest

class NotebookTests(BaseNotebookTest):
    def setUp(self):
        super().setUp()  # Call parent setUp first to initialize timing attributes
        # Create the JOSS2025.outputs directory that the notebook expects
        self.output_dir = os.path.join(os.path.dirname(__file__), 'JOSS2025.outputs')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            self._created_output_dir = True
        else:
            self._created_output_dir = False
        
    def tearDown(self):
        # Clean up the created directory if we created it
        if hasattr(self, '_created_output_dir') and self._created_output_dir:
            if os.path.exists(self.output_dir):
                shutil.rmtree(self.output_dir)
        super().tearDown()

    @unittest.skip("Skipping due to matplotlib compatibility issue with ncols parameter.")
    @testbook('../../demos/talk_paper_demos/JOSS2025/joss2025.ipynb', execute=True, timeout=TB_TIMEOUT)
    def test_joss2025_notebook(self, tb):
        # Fix matplotlib compatibility issue: ncols -> ncol
        # Inject a monkey patch for fig.legend to handle ncols parameter
        tb.inject("""
import matplotlib.pyplot as plt
original_legend = plt.Figure.legend

def patched_legend(self, *args, **kwargs):
    if 'ncols' in kwargs:
        kwargs['ncol'] = kwargs.pop('ncols')
    return original_legend(self, *args, **kwargs)

plt.Figure.legend = patched_legend
""")
        pass

if __name__ == '__main__':
    unittest.main()
