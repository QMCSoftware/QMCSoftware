import unittest
from testbook import testbook

class NotebookTests(unittest.TestCase):

    @testbook('../../demos/some_true_measures.ipynb', execute=True)
    def test_some_true_measures_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
