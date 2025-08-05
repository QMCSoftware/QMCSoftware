import unittest
from testbook import testbook

class NotebookTests(unittest.TestCase):

    @unittest.skip("API change: generalize parameter")
    @testbook('../../demos/dakota_genz.ipynb', execute=True)
    def test_dakota_genz_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
