import unittest
from testbook import testbook
from __init__ import TB_TIMEOUT, BaseNotebookTest, pip_install


class NotebookTests(BaseNotebookTest):

    def setUp(self):
        super().setUp()
        # Install required packages
        pip_install("scipy", "matplotlib")

    @testbook("../../demos/some_true_measures.ipynb", execute=True, timeout=TB_TIMEOUT)
    def test_some_true_measures_notebook(self, tb):
        pass


if __name__ == "__main__":
    unittest.main()
