import unittest
from testbook import testbook

class NotebookTests(unittest.TestCase):

    @testbook('../../demos/gaussian_diagnostics_demo.ipynb', execute=True)
    def test_gaussian_diagnostics_demo_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
