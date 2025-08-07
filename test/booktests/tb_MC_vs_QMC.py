import unittest
from testbook import testbook
from __init__ import TB_TIMEOUT, BaseNotebookTest

class NotebookTests(BaseNotebookTest):

    @testbook('../../demos/MC_vs_QMC.ipynb', execute=True, timeout=TB_TIMEOUT)
    def test_mc_vs_qmc_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
