import unittest
from testbook import testbook

class NotebookTests(unittest.TestCase):

    @testbook('../../demos/pydata.chi.2023.ipynb', execute=True)
    def test_pydata_chi_2023_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
