import unittest
from testbook import testbook
from __init__ import TB_TIMEOUT, BaseNotebookTest

@unittest.skip("Skipping Parsl Fest 2025 notebook tests Or it could result in infinite loop in CI workflow.")
class NotebookTests(BaseNotebookTest):

    @testbook('../../demos/_parsl_fest_2025.ipynb', execute=False, timeout=TB_TIMEOUT)
    def test__parsl_fest_2025_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
