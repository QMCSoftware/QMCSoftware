import unittest
from testbook import testbook

class NotebookTests(unittest.TestCase):

    @unittest.skip("Missing matplotlib style file")
    @testbook('../../demos/prob_failure_gp_ci.ipynb', execute=True)
    def test_prob_failure_gp_ci_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
