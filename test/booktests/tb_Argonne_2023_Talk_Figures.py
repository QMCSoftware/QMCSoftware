import unittest
from testbook import testbook

class NotebookTests(unittest.TestCase):

    @testbook('../../demos/talk_paper_demos/Argonne_Talk_2023_May/Argonne_2023_Talk_Figures.ipynb', execute=True)
    def test_argonne_talk_2023_figures_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
