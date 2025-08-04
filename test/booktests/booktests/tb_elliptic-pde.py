import unittest
from testbook import testbook

class NotebookTests(unittest.TestCase):

    @testbook('../../demos/elliptic-pde.ipynb', execute=True)
    def test_elliptic_pde_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
