import unittest
from testbook import testbook
from __init__ import TB_TIMEOUT, BaseNotebookTest

class NotebookTests(BaseNotebookTest):

    @unittest.skip("Runtime error")
    @testbook('../../demos/talk_paper_demos/Argonne_Talk_2023_May/Argonne_2023_Talk_Figures.ipynb', execute=True, timeout=TB_TIMEOUT)
    def test_argonne_talk_2023_figures_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
