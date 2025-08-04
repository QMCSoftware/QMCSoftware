import unittest
from testbook import testbook

class NotebookTests(unittest.TestCase):

    @testbook('../../demos/quasirandom_generators.ipynb', execute=True)
    def test_quasirandom_generators_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
