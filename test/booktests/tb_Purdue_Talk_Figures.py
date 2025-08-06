import unittest
from testbook import testbook
from __init__ import TB_TIMEOUT, BaseNotebookTest

class NotebookTests(BaseNotebookTest):

    @testbook('../../demos/talk_paper_demos/Purdue_Talk_2023_March/Purdue_Talk_Figures.ipynb', execute=True, timeout=TB_TIMEOUT)
    def test_purdue_talk_figures_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
