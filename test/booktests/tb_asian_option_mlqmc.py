import unittest, pytest
from testbook import testbook
from __init__ import TB_TIMEOUT, BaseNotebookTest

@pytest.mark.slow 
class NotebookTests(BaseNotebookTest):

    @testbook('../../demos/asian-option-mlqmc.ipynb', execute=True, timeout=TB_TIMEOUT)
    def test_asian_option_mlqmc_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
