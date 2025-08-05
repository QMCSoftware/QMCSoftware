import unittest
from testbook import testbook

class NotebookTests(unittest.TestCase):

    @unittest.skip("API change: generalize parameter")
    @testbook('../../demos/linear-scrambled-halton.ipynb', execute=True)
    def test_linear_scrambled_halton_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
