import unittest
from testbook import testbook
from __init__ import TB_TIMEOUT, BaseNotebookTest

class NotebookTests(BaseNotebookTest):

    @testbook('../../demos/talk_paper_demos/SorokinThesis2025/sorokin_thesis_2025.ipynb', execute=True, timeout=TB_TIMEOUT)
    def test_sorokin_thesis_2025_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
