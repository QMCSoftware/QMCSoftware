import unittest
from testbook import testbook

class NotebookTests(unittest.TestCase):

    @unittest.skip("Missing CSV file")
    @testbook('../../demos/MC_vs_QMC.ipynb', execute=True)
    def test_mc_vs_qmc_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
