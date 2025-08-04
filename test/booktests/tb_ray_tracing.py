import unittest
from testbook import testbook

class NotebookTests(unittest.TestCase):

    @testbook('../../demos/ray_tracing.ipynb', execute=True)
    def test_ray_tracing_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
