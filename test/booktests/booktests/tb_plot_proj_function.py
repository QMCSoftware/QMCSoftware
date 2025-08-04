import unittest
from testbook import testbook

class NotebookTests(unittest.TestCase):

    @testbook('../../demos/plot_proj_function.ipynb', execute=True)
    def test_plot_proj_function_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
