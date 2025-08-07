"""
Individual notebook test modules using testbook.
Each tb_*.py file tests a single demo notebook.
"""
import unittest, gc

TB_TIMEOUT = 3600

import psutil
import gc
import time

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

 