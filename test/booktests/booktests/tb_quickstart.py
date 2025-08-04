import unittest
from testbook import testbook

class NotebookTests(unittest.TestCase):

    @testbook('../../demos/quickstart.ipynb', execute=True)
    def test_quickstart_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
