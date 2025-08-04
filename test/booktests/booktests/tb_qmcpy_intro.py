import unittest
from testbook import testbook

class NotebookTests(unittest.TestCase):

    @testbook('../../demos/qmcpy_intro.ipynb', execute=True)
    def test_qmcpy_intro_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
