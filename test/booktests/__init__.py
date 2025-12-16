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

TB_TIMEOUT = 3600
subprocess.run(['pip', 'install', '-q', 'psutil', 'testbook', 'parsl'], check=False)

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
