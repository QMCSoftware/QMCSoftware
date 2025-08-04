import unittest
from testbook import testbook

class NotebookTests(unittest.TestCase):

    @testbook('../../demos/control_variates.ipynb', execute=True) 
    def test_control_variates_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
