import unittest
from testbook import testbook
from __init__ import TB_TIMEOUT, BaseNotebookTest

class NotebookTests(BaseNotebookTest):

    @testbook('../../demos/talk_paper_demos/ACMTOMS_Sorokin_2025/.ipynb_checkpoints/acm_toms_sorokin_2025-checkpoint.ipynb', execute=True, timeout=TB_TIMEOUT)
    def test_acm_toms_sorokin_2025_checkpoint_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
