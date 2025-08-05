import unittest
from testbook import testbook

class NotebookTests(unittest.TestCase):

    @unittest.skip("Times out (> 600s)")
    @testbook('../../demos/vectorized_qmc.ipynb', execute=True)
    def test_vectorized_qmc_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
