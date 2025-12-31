import unittest, pytest
from __init__ import TB_TIMEOUT, BaseNotebookTest


class NotebookTests(BaseNotebookTest):

    def test_gbm_examples_notebook(self):
        notebook_path, _ = self.locate_notebook('../../demos/GBM/gbm_examples.ipynb')
        symlinks_to_fix = ['config.py', 'data_util.py', 'latex_util.py',
                           'plot_util.py', 'qmcpy_util.py', 'quantlib_util.py']
        self.fix_gbm_symlinks(notebook_path, symlinks_to_fix)
        replacements = {"cf.is_debug = False": "cf.is_debug = True",}
        self.change_notebook_cells(notebook_path, replacements)
        self.run_notebook(notebook_path, timeout=TB_TIMEOUT)

if __name__ == '__main__':
    unittest.main()