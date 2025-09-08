import unittest
from testbook import testbook
from __init__ import TB_TIMEOUT, BaseNotebookTest

class NotebookTests(BaseNotebookTest):

    @testbook('../../demos/talk_paper_demos/pydata_chi_2023.ipynb', execute=True, timeout=TB_TIMEOUT)
    def test_pydata_chi_2023_notebook(self, tb):
        """Test that the notebook pydata.chi.2023.ipynb executes without errors."""
        pass

if __name__ == '__main__':
    unittest.main()
