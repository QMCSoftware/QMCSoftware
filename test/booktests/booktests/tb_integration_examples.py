import unittest
from testbook import testbook

class NotebookTests(unittest.TestCase):

    @testbook('../../demos/integration_examples.ipynb', execute=True)
    def test_integration_examples_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
