import unittest
from testbook import testbook

class NotebookTests(unittest.TestCase):

    @testbook('../../demos/importance_sampling.ipynb', execute=True)
    def test_importance_sampling_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
