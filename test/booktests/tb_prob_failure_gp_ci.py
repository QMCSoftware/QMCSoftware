import unittest
from testbook import testbook
from __init__ import TB_TIMEOUT, BaseNotebookTest

class NotebookTests(BaseNotebookTest):

    @unittest.skip("Requires external server - umbridge")
    @testbook('../../demos/talk_paper_demos/ProbFailureSorokinRao/prob_failure_gp_ci.ipynb', execute=True, timeout=TB_TIMEOUT)
    def test_prob_failure_gp_ci_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
