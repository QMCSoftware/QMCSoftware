import unittest
from testbook import testbook

class NotebookTests(unittest.TestCase):

    @testbook('../../demos/talk_paper_demos/MCQMC2022_Article_Figures/MCQMC2022_Article_Figures.ipynb', execute=True)
    def test_mcqmc2022_article_figures_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
