import unittest
from testbook import testbook

class NotebookTests(unittest.TestCase):

    @testbook('../../demos/lattice_random_generator.ipynb', execute=True)
    def test_lattice_random_generator_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
