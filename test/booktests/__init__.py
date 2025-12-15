"""
Individual notebook test modules using testbook.
Each tb_*.py file tests a single demo notebook.
"""
import unittest, gc

TB_TIMEOUT = 3600

import subprocess
subprocess.run(['pip', 'install', '-q', 'psutil', 'testbook', 'parsl'], check=False)

import psutil
import gc
import time
import os

class BaseNotebookTest(unittest.TestCase):
    """Base class for notebook tests with automatic memory cleanup"""
    
    def get_memory_usage(self):
        """Get current memory usage in GB"""
        process = psutil.Process()
        memory_info = process.memory_info()
        self.memory_gb = memory_info.rss / (1024**3)  # Convert bytes to GB

    def setUp(self):
        """Clean up before each test"""
        self.start_time = time.time()
        gc.collect()
    
    def tearDown(self):
        """Clean up after each test"""
        end_time = time.time()  
        self.get_memory_usage()
        print(f"    Memory used: {self.memory_gb:.2f} GB.  Test time: {end_time - self.start_time:.2f} s")
        gc.collect()

    # -- Shared helpers for notebook tests -------------------------------------------------
    def locate_notebook(self, rel_path):
        """Return absolute notebook path and its directory. Skip test if missing."""
        notebook_path = os.path.join(os.path.dirname(__file__), rel_path)
        if not os.path.exists(notebook_path):
            self.skipTest(f"Notebook not found at {notebook_path}")
        notebook_path = os.path.abspath(notebook_path)
        return notebook_path, os.path.dirname(notebook_path)

    def fix_gbm_symlinks(self, notebook_dir):
        """Fix or create symlinks inside a GBM demo notebook directory."""
        code_dir = os.path.join(notebook_dir, 'gbm_code')
        symlinks_to_fix = ['config.py', 'data_util.py', 'latex_util.py', 'plot_util.py', 
                          'qmcpy_util.py', 'quantlib_util.py']

        for module in symlinks_to_fix:
            symlink_path = os.path.join(notebook_dir, module)
            target_path = os.path.join(code_dir, module)

            if os.path.islink(symlink_path):
                link_target = os.readlink(symlink_path)
                if link_target.startswith('code/') or not os.path.exists(symlink_path):
                    os.remove(symlink_path)
                    os.symlink(f'gbm_code/{module}', symlink_path)
            elif not os.path.exists(symlink_path) and os.path.exists(target_path):
                os.symlink(f'gbm_code/{module}', symlink_path)

    def run_notebook(self, notebook_path, notebook_dir, timeout=TB_TIMEOUT):
        """Execute a notebook using testbook from its directory."""
        from testbook import testbook

        original_dir = os.getcwd()
        try:
            os.chdir(notebook_dir)
            with testbook(os.path.basename(notebook_path), execute=True, timeout=timeout):
                pass
        finally:
            os.chdir(original_dir)

 