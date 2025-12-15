import unittest, os
from testbook import testbook
from __init__ import TB_TIMEOUT, BaseNotebookTest

def fix_gbm_symlinks(notebook_dir):
    """Fix or create symlinks inside a GBM demo notebook directory."""
    code_dir = os.path.join(notebook_dir, 'gbm_code')
    symlinks_to_fix = ['config.py', 'data_util.py', 'latex_util.py', 'plot_util.py', 
                        'qmcpy_util.py', 'quantlib_util.py']

    for module in symlinks_to_fix:
        symlink_path = os.path.join(notebook_dir, module)
        target_path = os.path.join(code_dir, module)

        if os.path.islink(symlink_path):
            if link_target.startswith('code/') or not os.path.exists(symlink_path):
                os.remove(symlink_path)
                os.symlink(f'gbm_code/{module}', symlink_path)
        elif not os.path.exists(symlink_path) and os.path.exists(target_path):
            os.symlink(f'gbm_code/{module}', symlink_path)

    
def run_notebook(notebook_path, notebook_dir, timeout=TB_TIMEOUT, change_value=False, 
                value='cf.is_debug = False', new_value='cf.is_debug = True'):
    """Execute a notebook using testbook from its directory.
    """
    original_dir = os.getcwd()
    try:
        os.chdir(notebook_dir)
        if change_value:
            with testbook(os.path.basename(notebook_path), execute=False) as tb:
                for cell in tb.nb.cells:
                    if cell.get('cell_type') == 'code':
                        src = cell.get('source', '')
                        if 'cf.is_debug = False' in src:
                            cell['source'] = src.replace(value, new_value) 
                tb.execute()
        else:
            with testbook(os.path.basename(notebook_path), execute=True, timeout=timeout):
                pass
    finally:
        os.chdir(original_dir)

class NotebookTests(BaseNotebookTest):

    def test_gbm_examples_notebook(self):
        notebook_path, notebook_dir = self.locate_notebook('../../demos/GBM/gbm_examples.ipynb')
        fix_gbm_symlinks(notebook_dir)
        # Toggle code cell [3] cf.is_debug -> True before executing
        run_notebook(notebook_path, notebook_dir, change_value=True)

if __name__ == '__main__':
    unittest.main()