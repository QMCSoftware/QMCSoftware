import unittest
from testbook import testbook
from __init__ import TB_TIMEOUT, BaseNotebookTest

class NotebookTests(BaseNotebookTest):

    @testbook('../../demos/talk_paper_demos/Sorokin_random_LD_seq_QMC_fast_kernel_methods_2026/Sorokin_random_LD_seq_QMC_fast_kernel_methods_2026.ipynb', execute=True, timeout=TB_TIMEOUT)
    def test_Sorokin_random_LD_seq_QMC_fast_kernel_methods_2026_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
