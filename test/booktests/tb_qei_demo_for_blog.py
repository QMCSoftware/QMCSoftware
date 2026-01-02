import unittest, pytest
from testbook import testbook
from __init__ import BaseNotebookTest

class NotebookTests(BaseNotebookTest):

    def test_qei_demo_for_blog_notebook(self):

        notebook_path, _ = self.locate_notebook('../../demos/qei-demo-for-blog.ipynb')
        replacements = {
            "abs_tol=5e-7)":"abs_tol=1e-2)",
            "[2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]":"[1e-2]",
            "[5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]": "[1e-2]",
                        }
        self.run_notebook(notebook_path, replacements)

if __name__ == '__main__':
    unittest.main()
