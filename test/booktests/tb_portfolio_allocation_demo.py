import unittest
from testbook import testbook
from __init__ import TB_TIMEOUT, BaseNotebookTest

class NotebookTests(BaseNotebookTest):

    @testbook('../../demos/portfolio/portfolio_allocation_demo.ipynb', execute=True, timeout=TB_TIMEOUT)
    def test_portfolio_allocation_demo_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
