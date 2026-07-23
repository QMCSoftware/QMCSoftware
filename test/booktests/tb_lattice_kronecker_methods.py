import unittest
from __init__ import BaseNotebookTest


class NotebookTests(BaseNotebookTest):

    def test_lattice_kronecker_methods_notebook(self):
        # Keep enough lattice candidates for the reduced dimension: dim <= n / 4.
        replacements = {
            "dim = 64": "dim = 8",
            "n = 2**20": "n = 2**5",
            "searchsize = 25": "searchsize = 4",
        }
        self.run_notebook(
            "../../demos/lattice_kronecker_methods.ipynb",
            replacements=replacements,
        )


if __name__ == '__main__':
    unittest.main()
