import unittest
from testbook import testbook

class NotebookTests(unittest.TestCase):

    @testbook('../../demos/lebesgue_integration.ipynb', execute=True)
    def test_lebesgue_integration_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
