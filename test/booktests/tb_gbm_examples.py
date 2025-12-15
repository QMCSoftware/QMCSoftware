import os
import sys
import unittest
from testbook import testbook

class NotebookTests(unittest.TestCase):
    
    def test_gbm_examples_notebook(self):
        # Find the notebook path
        notebook_path = '../../demos/GBM/gbm_examples.ipynb'
        
        if not os.path.exists(notebook_path):
            self.skipTest(f"Notebook not found at {notebook_path}")
        
        notebook_path = os.path.abspath(notebook_path)
        notebook_dir = os.path.dirname(notebook_path)
        
        # Fix symlinks if they're broken (pointing to non-existent 'code/' instead of 'gbm_code/')
        code_dir = os.path.join(notebook_dir, 'gbm_code')
        symlinks_to_fix = ['config.py', 'data_util.py', 'latex_util.py', 'plot_util.py', 
                          'qmcpy_util.py', 'quantlib_util.py']
        
        for module in symlinks_to_fix:
            symlink_path = os.path.join(notebook_dir, module)
            target_path = os.path.join(code_dir, module)
            
            # Check if symlink exists and is broken or points to wrong location
            if os.path.islink(symlink_path):
                link_target = os.readlink(symlink_path)
                # If link points to 'code/' or is broken, fix it
                if link_target.startswith('code/') or not os.path.exists(symlink_path):
                    os.remove(symlink_path)
                    os.symlink(f'gbm_code/{module}', symlink_path)
            elif not os.path.exists(symlink_path) and os.path.exists(target_path):
                # Create symlink if it doesn't exist but target does
                os.symlink(f'gbm_code/{module}', symlink_path)
        
        # Execute from the notebook's directory
        original_dir = os.getcwd()
        try:
            os.chdir(notebook_dir)
            with testbook(os.path.basename(notebook_path), execute=True, timeout=600):
                pass
        finally:
            os.chdir(original_dir)

if __name__ == '__main__':
    unittest.main()