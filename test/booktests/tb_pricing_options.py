import unittest
from testbook import testbook
from base_notebook_test import TB_TIMEOUT, BaseNotebookTest


class NotebookTests(BaseNotebookTest):
    @testbook("../../demos/pricing_options.ipynb", execute=True, timeout=TB_TIMEOUT)
    def test_pricing_options_notebook(self, tb):
        pass


if __name__ == "__main__":
    unittest.main()
