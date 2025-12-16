import unittest
from __init__ import TB_TIMEOUT, BaseNotebookTest, fix_symlinks, run_notebook

def fix_gbm_symlinks(notebook_dir):
    """Fix or create symlinks inside a GBM demo notebook directory."""
    symlinks_to_fix = ['config.py', 'data_util.py', 'latex_util.py', 'plot_util.py',
                        'qmcpy_util.py', 'quantlib_util.py']
    fix_symlinks(notebook_dir, 'gbm_code', symlinks_to_fix)

class NotebookTests(BaseNotebookTest):

    def test_gbm_examples_notebook(self):
        notebook_path, notebook_dir = self.locate_notebook('../../demos/GBM/gbm_examples.ipynb')
        fix_gbm_symlinks(notebook_dir)
        # Toggle code cell [3] cf.is_debug -> True before executing
        run_notebook(notebook_path, notebook_dir, change_value=True,
                     value='cf.is_debug = False', new_value='cf.is_debug = True')

if __name__ == '__main__':
    unittest.main()