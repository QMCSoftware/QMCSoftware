import unittest, pytest
from testbook import testbook
from __init__ import BaseNotebookTest

class NotebookTests(BaseNotebookTest):

    def test_qei_demo_for_blog_notebook(self):

        notebook_path, _ = self.locate_notebook('../../demos/qei-demo-for-blog.ipynb')
        replacements = {
            "abs_tol=5e-7)": "abs_tol=1e-1)",
            "[2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]": "[1e-1]",
            "[5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]": "[1e-1]",
            # Reduce large sample / iteration counts
            "n_samples=10000": "n_samples=10",
            "n_samples = 10000": "n_samples = 10",
            "num_samples=10000": "num_samples=10",
            "N = 10000": "N = 10",
            "N=10000": "N=10",
            "n_mc=100000": "n_mc=10",
            "n_draws=100000": "n_draws=10",
            "n_iter=1000": "n_iter=10",
            "n_iters=1000": "n_iters=10",
            "max_iter=1000": "max_iter=10",
            "range(1000)": "range(4)",
            "range(10000)": "range(4)",
            "trials = 25": "trials = 1",
            # Reduce plotting / interactive delays
            "plt.show()": "# plt.show()",
            "time.sleep(1)": "time.sleep(0.1)",
        }
        self.run_notebook(notebook_path, replacements)

if __name__ == '__main__':
    unittest.main()
