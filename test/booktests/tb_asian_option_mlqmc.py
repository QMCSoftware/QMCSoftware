import unittest, pytest
from __init__ import BaseNotebookTest

@pytest.mark.slow
class NotebookTests(BaseNotebookTest):

    def test_asian_option_mlqmc_notebook(self, tb):
        notebook_path, _ = self.locate_notebook('../../demos/asian-option-mlqmc.ipynb')
        replacements = {"for level in range(5):":"for level in range(2)",
                        "abs_tol=5e-3":"abs_tol=1e-2",
                        "tolerances = 5*np.logspace(-1, -3, num=5)":"tolerances = 5*np.logspace(-1, -2, num=2)",
                        "for method in range(4):":"for method in range(2):",
                        }
        self.run_notebook(notebook_path, replacements)

if __name__ == '__main__':
    unittest.main()
