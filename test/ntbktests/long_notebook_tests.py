""" Unit tests for long-running demo notebooks execution using testbook """
import unittest
from testbook import testbook

class LongNotebookTests(unittest.TestCase):
    """Test class for long-running demo notebooks (> 30 seconds typical execution)"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.demos_path = '../../demos/'
    
    @testbook('../../demos/control_variates.ipynb', execute=True) 
    def test_control_variates_notebook(self, tb):
        pass
    
    @testbook('../../demos/elliptic-pde.ipynb', execute=True)
    def test_elliptic_pde_notebook(self, tb):
        pass
    
    @testbook('../../demos/nei_demo.ipynb', execute=True)
    def test_nei_demo_notebook(self, tb):
        pass

    @testbook('../../demos/qei-demo-for-blog.ipynb', execute=True)
    def test_qei_demo_blog_notebook(self, tb):
        pass
    
    @testbook('../../demos/ray_tracing.ipynb', execute=True)
    def test_ray_tracing_notebook(self, tb):
        pass
    

if __name__ == '__main__':
    unittest.main()
    # python -m pytest long_notebook_tests.py 
    # ============= 5 passed, 1 warning in 118.43s (0:01:58)
