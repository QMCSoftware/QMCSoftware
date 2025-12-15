import unittest
from __init__ import BaseNotebookTest
from tb_gbm_examples import fix_gbm_symlinks, run_notebook
class NotebookTests(BaseNotebookTest):
    
    def test_gbm_demo_notebook(self):
        notebook_path, notebook_dir = self.locate_notebook('../../demos/GBM/gbm_demo.ipynb')
        fix_gbm_symlinks(notebook_dir)
        self.execute_notebook_file(notebook_path, timeout=600)
        # Toggle code cell [3] cf.is_debug -> True before executing
        run_notebook(notebook_path, notebook_dir, change_value=True, timeout=600)
        
        
if __name__ == '__main__':
    unittest.main()