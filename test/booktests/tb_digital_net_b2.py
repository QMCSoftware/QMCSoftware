import unittest, pytest
from testbook import testbook
from __init__ import TB_TIMEOUT, BaseNotebookTest

@pytest.mark.slow
class NotebookTests(BaseNotebookTest):

    @testbook('../../demos/digital_net_b2.ipynb', execute=True, timeout=TB_TIMEOUT)
    def test_digital_net_b2_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
