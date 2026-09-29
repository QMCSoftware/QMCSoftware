import unittest, pytest
from __init__ import BaseNotebookTest


@pytest.mark.slow
class NotebookTests(BaseNotebookTest):

    notebook_path = f"../../demos/elliptic-pde.ipynb"

    def test_elliptic_pde_notebook(self):
        replacements = {
            "plot_convergence(execute_convergence_test": "#plot_convergence(execute_convergence_test",
        }
        self.run_notebook(self.notebook_path, replacements)


if __name__ == "__main__":
    unittest.main()
