import unittest
from __init__ import TB_TIMEOUT, BaseNotebookTest
from tb_gbm_examples import fix_gbm_symlinks, run_notebook

@unittest.skip("Skipping GBM demo notebook test") # Error: https://github.com/QMCSoftware/QMCSoftware/actions/runs/20248843941/job/58135823467#step:15:206
class NotebookTests(BaseNotebookTest):
    
    def test_gbm_demo_notebook(self):
        notebook_path, notebook_dir = self.locate_notebook('../../demos/GBM/gbm_demo.ipynb')
        symlinks_to_fix = ['config.py', 'data_util.py', 'latex_util.py', 'plot_util.py', 
                            'qmcpy_util.py', 'quantlib_util.py']
        self.fix_gbm_symlinks(notebook_path, symlinks_to_fix=symlinks_to_fix)
        # Toggle code cell [3] cf.is_debug -> True, then execute
        self.change_notebook_cells(notebook_path, 
                                   replacements={"cf.is_debug = False": "cf.is_debug = True"})
        self.run_notebook(notebook_path, timeout=TB_TIMEOUT)
        
        
if __name__ == '__main__':
    unittest.main()