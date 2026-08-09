import unittest
from pathlib import Path

from testbook import testbook
from __init__ import TB_TIMEOUT, BaseNotebookTest

NOTEBOOK = Path(__file__).resolve().parents[2] / "demos" / "acceptance_rejection.ipynb"


class NotebookTests(BaseNotebookTest):

    @testbook(NOTEBOOK.as_posix(), execute=True, timeout=TB_TIMEOUT)
    def test_acceptance_rejection_notebook(self, tb):
        pass


if __name__ == "__main__":
    unittest.main()
