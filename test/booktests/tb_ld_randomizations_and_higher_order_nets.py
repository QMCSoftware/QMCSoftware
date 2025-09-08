import unittest
from testbook import testbook
from __init__ import TB_TIMEOUT, BaseNotebookTest

class NotebookTests(BaseNotebookTest):

    @testbook('../../demos/ld_randomizations_and_higher_order_nets.ipynb', execute=True, timeout=TB_TIMEOUT)
    def test_ld_randomizations_and_higher_order_nets_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
