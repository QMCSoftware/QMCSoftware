import unittest
from testbook import testbook
from __init__ import TB_TIMEOUT, BaseNotebookTest

class NotebookTests(BaseNotebookTest):

    @testbook('../../demos/plot_proj_function.ipynb', execute=True, timeout=TB_TIMEOUT)
    def test_plot_proj_function_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
