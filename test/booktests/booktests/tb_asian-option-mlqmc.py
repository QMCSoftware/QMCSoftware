import unittest
from testbook import testbook

class NotebookTests(unittest.TestCase):

    @testbook('../../demos/asian-option-mlqmc.ipynb', execute=True)
    def test_asian_option_mlqmc_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
