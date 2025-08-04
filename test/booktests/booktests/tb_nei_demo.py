import unittest
from testbook import testbook

class NotebookTests(unittest.TestCase):

    @testbook('../../demos/nei_demo.ipynb', execute=True)
    def test_nei_demo_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
