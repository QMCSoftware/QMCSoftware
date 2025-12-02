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
import matplotlib.pyplot as plt
import os
import glob

# Simple global counters for progress display
_NOTEBOOK_TEST_INDEX = 0
_TOTAL_NOTEBOOK_TESTS = int(os.environ.get("TOTAL_NOTEBOOK_TESTS", "0") or "0")


class BaseNotebookTest(unittest.TestCase):
    """Base class for notebook tests with automatic memory cleanup"""
    
    def get_memory_usage(self):
        """
        Approximate memory usage as RSS of this process plus all children.
        Stores result in self.memory_gb (GB).
        """
        try:
            proc = psutil.Process(os.getpid())
        except psutil.Error:
            self.memory_gb = 0.0
            return

        rss = 0
        try:
            rss += proc.memory_info().rss
        except psutil.Error:
            pass

        # Include all child processes (e.g., the Jupyter kernel testbook uses)
        for child in proc.children(recursive=True):
            try:
                rss += child.memory_info().rss
            except psutil.Error:
                continue

        # Convert bytes → GB
        self.memory_gb = rss / (1024 ** 3)

    def setUp(self):
        """Clean up before each test"""
        self.start_time = time.time()
        gc.collect()
    
    def tearDown(self):
        """Clean up after each test"""
        global _NOTEBOOK_TEST_INDEX, _TOTAL_NOTEBOOK_TESTS
        end_time = time.time()
        self.get_memory_usage()

        # Increment index
        _NOTEBOOK_TEST_INDEX += 1

        # Use "?" if TOTAL is not known for some reason
        total = _TOTAL_NOTEBOOK_TESTS or "?"

        print(
            f"    [{_NOTEBOOK_TEST_INDEX}/{total}] "
            f"Memory used: {self.memory_gb:.2f} GB.  "
            f"Test time: {end_time - self.start_time:.2f} s"
        )

        # --- New cleanup logic to reduce memory / disk usage ---

        # Close all matplotlib figures so they don't accumulate in memory
        plt.close('all')

        # Remove temporary figure / log files created by notebooks in this dir
        patterns = [
            "*.eps", "*.pdf", "*.png", "*.jpg", "*.jpeg",
            "*.svg", "*.log", "*.txt"
        ]
        for pattern in patterns:
            for path in glob.glob(pattern):
                try:
                    os.remove(path)
                except OSError:
                    # Ignore files we can't delete (permissions, race conditions, etc.)
                    pass

        # Existing GC call
        gc.collect()