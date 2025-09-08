import unittest
from testbook import testbook
from __init__ import TB_TIMEOUT, BaseNotebookTest

class NotebookTests(BaseNotebookTest):
    @testbook('../../demos/talk_paper_demos/JOSS2025/joss2025.ipynb', execute=True, timeout=TB_TIMEOUT)
    def test_joss2025_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
