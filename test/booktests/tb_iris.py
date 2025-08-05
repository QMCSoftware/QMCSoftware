import unittest
from testbook import testbook

class NotebookTests(unittest.TestCase):

    @unittest.skip("Missing skopt dependency")
    @testbook('../../demos/iris.ipynb', execute=True)
    def test_iris_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
