import unittest
from testbook import testbook

class NotebookTests(unittest.TestCase):

    @testbook('../../demos/sample_scatter_plots.ipynb', execute=True)
    def test_sample_scatter_plots_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
