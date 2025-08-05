import unittest
from testbook import testbook

class NotebookTests(unittest.TestCase):

    #@unittest.skip("Times out (> 1200s)")
    @testbook('../../demos/asian-option-mlqmc.ipynb', execute=True, timeout=1500)
    def test_asian_option_mlqmc_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
