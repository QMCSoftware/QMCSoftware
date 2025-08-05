import unittest
from testbook import testbook

class NotebookTests(unittest.TestCase):

    @unittest.skip("Requires external server")
    @testbook('../../demos/umbridge.ipynb', execute=True)
    def test_umbridge_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
