import unittest
from testbook import testbook

class NotebookTests(unittest.TestCase):

    @unittest.skip("Times out (> 60s) - Bayesian inference computationally intensive")
    @testbook('../../demos/vectorized_qmc_bayes.ipynb', execute=True)
    def test_vectorized_qmc_bayes_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
