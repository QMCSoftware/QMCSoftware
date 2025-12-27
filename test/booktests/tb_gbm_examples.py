import unittest, pytest
from __init__ import TB_TIMEOUT, BaseNotebookTest

@pytest.mark.slow 
class NotebookTests(BaseNotebookTest):

    def test_gbm_examples_notebook(self):
        symlinks_to_fix = ['config.py', 'data_util.py', 'latex_util.py', 'plot_util.py', 
                            'qmcpy_util.py', 'quantlib_util.py']
        notebook_path, notebook_dir = self.locate_notebook('../../demos/GBM/gbm_examples.ipynb')
        self.fix_gbm_symlinks(notebook_dir, symlinks_to_fix=symlinks_to_fix)
        # Toggle code cell [3] cf.is_debug -> True before executing
        self.change_notebook_cells(notebook_path, 
                                   replacements={"cf.is_debug = False": "cf.is_debug = True"})
        self.run_notebook(notebook_path)

if __name__ == '__main__':
    unittest.main()