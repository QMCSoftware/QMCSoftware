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
import nbformat
import matplotlib

matplotlib.rcParams["text.usetex"] = False  # Disable LaTeX

TB_TIMEOUT = 3600
subprocess.run(["pip", "install", "-q", "psutil", "testbook", "parsl"], check=False)


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
        print(
            f"    Memory used: {self.memory_gb:.2f} GB.  Test time: {end_time - self.start_time:.2f} s"
        )
        gc.collect()

    # -- Shared helpers for notebook tests -------------------------------------------------
    def locate_notebook(self, rel_path):
        """Return absolute notebook path and its directory. Skip test if missing."""
        notebook_path = os.path.join(os.path.dirname(__file__), rel_path)
        if not os.path.exists(notebook_path):
            self.skipTest(f"Notebook not found at {notebook_path}")
        notebook_path = os.path.abspath(notebook_path)
        return notebook_path, os.path.dirname(notebook_path)

    def fix_gbm_symlinks(self, notebook_dir, symlinks_to_fix=None):
        """Fix or create symlinks inside a demo notebook directory."""
        code_dir = os.path.join(notebook_dir, "gbm_code")
        if not symlinks_to_fix:
            return
        for module in symlinks_to_fix:
            symlink_path = os.path.join(notebook_dir, module)
            target_path = os.path.join(code_dir, module)
            if os.path.islink(symlink_path):
                link_target = os.readlink(symlink_path)
                if link_target.startswith("code/") or not os.path.exists(symlink_path):
                    os.remove(symlink_path)
                    os.symlink(f"gbm_code/{module}", symlink_path)
            elif not os.path.exists(symlink_path) and os.path.exists(target_path):
                os.symlink(f"gbm_code/{module}", symlink_path)

    def run_notebook(
        self, notebook_path, replacements=None, is_overwrite=False, timeout=TB_TIMEOUT,
        stop_at_pattern=None, skip_patterns=None
    ):
        """Execute a notebook file using nbconvert's ExecutePreprocessor.

        Args:
            notebook_path: Path to the notebook file
            timeout: Execution timeout in seconds
            replacements: Optional dict of {old_str: new_str} to apply to code cells in memory
            is_overwrite: If True, overwrite the notebook file with modified cells
            stop_at_pattern: Optional string pattern - if provided, uses `testbook` to stop execution
                           at the first cell containing this pattern
            skip_patterns: Optional list of string patterns - cells containing any of these patterns will be skipped
        """        
        # If stop_at_pattern or skip_patterns is provided, use testbook for selective execution
        if stop_at_pattern or skip_patterns:
            with open(notebook_path) as f:
                nb = nbformat.read(f, as_version=4)
            
            # Apply replacements if provided
            if replacements:
                for cell in nb.cells:
                    if cell.get('cell_type') == 'code':
                        src = cell.get('source', '')
                        for old, new in replacements.items():
                            if old in src:
                                src = src.replace(old, new)
                        cell['source'] = src
            
            if is_overwrite:
                nbformat.write(nb, notebook_path)
                print(f"Notebook {notebook_path} overwritten with modified cells.")
            
            # Write modified notebook to a temp file so testbook uses the replacements
            # Use a hidden file name to avoid polluting the directory
            import uuid
            notebook_dir = os.path.dirname(notebook_path)
            temp_path = os.path.join(notebook_dir, f'.tmp_test_{uuid.uuid4().hex[:8]}.ipynb')
            try:
                nbformat.write(nb, temp_path)
                
                # Use testbook to execute cells until we hit the stop pattern
                # Execute from the notebook's directory to ensure imports work
                original_cwd = os.getcwd()
                skip_patterns = skip_patterns or []
                try:
                    os.chdir(notebook_dir)
                    with testbook(temp_path, timeout=timeout, execute=False) as tb:
                        for i, cell in enumerate(tb.cells):
                            if cell.cell_type == 'code':
                                if stop_at_pattern and stop_at_pattern in cell.source:
                                    print(f"Stopping execution at cell {i}: found pattern '{stop_at_pattern}'")
                                    break
                                # Check if this cell should be skipped
                                should_skip = any(pattern in cell.source for pattern in skip_patterns)
                                if should_skip:
                                    print(f"Skipping cell {i}: matched skip pattern")
                                    continue
                                try:
                                    tb.execute_cell(i)
                                except Exception as e:
                                    raise RuntimeError(f"Error executing cell {i} in {notebook_path}: {e}")
                finally:
                    os.chdir(original_cwd)
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            return
        
        # Otherwise use the original ExecutePreprocessor method
        from nbconvert.preprocessors import ExecutePreprocessor

        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)

        # Apply replacements in memory if provided
        if replacements:
            for cell in nb.cells:
                if cell.get("cell_type") == "code":
                    src = cell.get("source", "")
                    for old, new in replacements.items():
                        if old in src:
                            src = src.replace(old, new)
                    cell["source"] = src

        if is_overwrite:
            nbformat.write(nb, notebook_path)
            print(f"Notebook {notebook_path} overwritten with modified cells.")

        ep = ExecutePreprocessor(timeout=timeout, kernel_name="python3")
        try:
            ep.preprocess(nb, {"metadata": {"path": os.path.dirname(notebook_path)}})
        except Exception as e:
            raise RuntimeError(f"Error executing the notebook {notebook_path}: {e}")
