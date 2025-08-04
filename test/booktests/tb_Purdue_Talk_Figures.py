import unittest
from testbook import testbook

class NotebookTests(unittest.TestCase):

    @testbook('../../demos/talk_paper_demos/Purdue_Talk_2023_March/Purdue_Talk_Figures.ipynb', execute=True)
    def test_purdue_talk_figures_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
