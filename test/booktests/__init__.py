"""
Individual notebook test modules using testbook.
Each tb_*.py file tests a single demo notebook.
"""
import unittest, gc
import psutil
import gc
import time
import os
import subprocess
from testbook import testbook

TB_TIMEOUT = 3600
subprocess.run(['pip', 'install', '-q', 'psutil', 'testbook', 'parsl'], check=False)


def fix_symlinks(notebook_dir, code_subdir, symlinks_to_fix):
    """Fix or create symlinks inside a demo notebook directory.
    
    Args:
        notebook_dir: Directory containing the notebook
        code_subdir: Subdirectory name containing the code files (e.g., 'gbm_code')
        symlinks_to_fix: List of module filenames to create symlinks for
    """
    code_dir = os.path.join(notebook_dir, code_subdir)
    
    for module in symlinks_to_fix:
        symlink_path = os.path.join(notebook_dir, module)
        target_path = os.path.join(code_dir, module)

        if os.path.islink(symlink_path):
            link_target = os.readlink(symlink_path)
            if link_target.startswith('code/') or not os.path.exists(symlink_path):
                os.remove(symlink_path)
                os.symlink(f'{code_subdir}/{module}', symlink_path)
        elif not os.path.exists(symlink_path) and os.path.exists(target_path):
            os.symlink(f'{code_subdir}/{module}', symlink_path)


def run_notebook(notebook_path, notebook_dir, timeout=TB_TIMEOUT,
                change_value=False, value=None, new_value=None):
    """Execute a notebook using testbook from its directory.
    
    Args:
        notebook_path: Path to the notebook file
        notebook_dir: Directory to execute the notebook from
        timeout: Maximum execution time in seconds
        change_value: Whether to replace values in the notebook before executing
        value: String or list of strings to find and replace
        new_value: String or list of strings to replace with (must match length of value if lists)
    """
    original_dir = os.getcwd()
    try:
        os.chdir(notebook_dir)
        
        if change_value:
            if value is None or new_value is None:
                raise ValueError("value and new_value must be provided when change_value=True")
            
            # Convert single values to lists for uniform processing
            values = value if isinstance(value, list) else [value]
            new_values = new_value if isinstance(new_value, list) else [new_value]
            
            if len(values) != len(new_values):
                raise ValueError("value and new_value lists must have the same length")
            
            with testbook(os.path.basename(notebook_path), execute=False, timeout=timeout) as tb:
                for old_val, new_val in zip(values, new_values):
                    for cell in tb.nb.cells:
                        if cell.get('cell_type') == 'code':
                            src = cell.get('source', '')
                            if old_val in src:
                                cell['source'] = src.replace(old_val, new_val)
                tb.execute()
        else:
            with testbook(os.path.basename(notebook_path), execute=True, timeout=timeout):
                pass
    finally:
        os.chdir(original_dir)

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

    def execute_notebook_file(self, notebook_path, timeout=TB_TIMEOUT):
        """Execute a notebook file using nbconvert's ExecutePreprocessor."""
        import nbformat
        from nbconvert.preprocessors import ExecutePreprocessor

        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)

        ep = ExecutePreprocessor(timeout=timeout, kernel_name='python3')

        try:
            ep.preprocess(nb, {'metadata': {'path': os.path.dirname(notebook_path)}})
        except Exception as e:
            raise RuntimeError(f"Error executing the notebook {notebook_path}: {e}")
