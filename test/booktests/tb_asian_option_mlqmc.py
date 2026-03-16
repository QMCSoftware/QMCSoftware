import unittest, pytest
from __init__ import BaseNotebookTest


class NotebookTests(BaseNotebookTest):

    def test_asian_option_mlqmc_notebook(self):
        notebook_path, _ = self.locate_notebook("../../demos/asian-option-mlqmc.ipynb")
        replacements = {
            "for level in range(5):": "for level in range(2):",
            "abs_tol=5e-3": "abs_tol=1e-2",
            "tolerances = 5*np.logspace(-1, -3, num=5)": "tolerances = 5*np.logspace(-1, -2, num=2)",
            "for method in range(4):": "for method in range(2):",
            'plt.plot(tolerances, avg_time[2], label="MLQMC")': '#plt.plot(tolerances, avg_time[2], label="MLQMC")',
            'plt.plot(tolerances, avg_time[3], label="continuation MLQMC")': '#plt.plot(tolerances, avg_time[3], label="continuation MLQMC")',
            "plt.subplot(2,2,3); plt.bar(range(15), max_levels[2],": "#plt.subplot(2,2,3); plt.bar(range(15), max_levels[2],",
            "plt.subplot(2,2,4); plt.bar(range(15), max_levels[3]": "#plt.subplot(2,2,4); plt.bar(range(15), max_levels[3]",
        }
        self.run_notebook(notebook_path, replacements)


if __name__ == "__main__":
    unittest.main()
