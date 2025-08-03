""" Unit tests for fast-running demo notebooks execution using testbook """
import unittest
from testbook import testbook

class FastNotebookTests(unittest.TestCase):
    """Test class for fast-running demo notebooks (< 30 seconds typical execution)"""
    
    def setUp(self):
        self.demos_path = '../../demos/'
    
    @testbook('../../demos/quickstart.ipynb', execute=True)
    def test_quickstart_notebook(self, tb):
        pass
    
    @testbook('../../demos/qmcpy_intro.ipynb', execute=True)
    def test_qmcpy_intro_notebook(self, tb):
        pass
    
    @testbook('../../demos/integration_examples.ipynb', execute=True)
    def test_integration_examples_notebook(self, tb):
        pass
    
    @testbook('../../demos/lebesgue_integration.ipynb', execute=True)
    def test_lebesgue_integration_notebook(self, tb):
        pass
    
    @testbook('../../demos/plot_proj_function.ipynb', execute=True)
    def test_plot_proj_function_notebook(self, tb):
        pass
    
    @testbook('../../demos/some_true_measures.ipynb', execute=True)
    def test_some_true_measures_notebook(self, tb):
        pass


if __name__ == '__main__':
    unittest.main()
    # python -m pytest fast_notebook_tests.py 
    # ====================== 6 passed in 20.38s =============