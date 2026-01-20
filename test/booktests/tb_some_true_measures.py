import unittest
import subprocess
from testbook import testbook
from base_notebook_test import TB_TIMEOUT, BaseNotebookTest


class NotebookTests(BaseNotebookTest):

    def setUp(self):
        super().setUp()
        # Install required packages
        subprocess.run(["pip", "install", "-q", "scipy", "matplotlib"], check=False)

    @testbook("../../demos/some_true_measures.ipynb", execute=True, timeout=TB_TIMEOUT)
    def test_some_true_measures_notebook(self, tb):
        pass


if __name__ == "__main__":
    unittest.main()
