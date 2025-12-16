import unittest
from __init__ import TB_TIMEOUT, BaseNotebookTest, run_notebook
from tb_gbm_examples import fix_gbm_symlinks

@unittest.skip("Skipping GBM demo notebook test") # Error: https://github.com/QMCSoftware/QMCSoftware/actions/runs/20248843941/job/58135823467#step:15:206
class NotebookTests(BaseNotebookTest):
    
    def test_gbm_demo_notebook(self):
        notebook_path, notebook_dir = self.locate_notebook('../../demos/GBM/gbm_demo.ipynb')
        fix_gbm_symlinks(notebook_dir)
        self.execute_notebook_file(notebook_path, timeout=TB_TIMEOUT)
        # Toggle code cell [3] cf.is_debug -> True before executing
        run_notebook(notebook_path, notebook_dir, change_value=True, timeout=TB_TIMEOUT,
                     value='cf.is_debug = False', new_value='cf.is_debug = True')
        
        
if __name__ == '__main__':
    unittest.main()