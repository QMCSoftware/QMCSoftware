import unittest
from __init__ import BaseNotebookTest


class NotebookTests(BaseNotebookTest):

    def test_ray_tracing_notebook(self):
        notebook_path, _ = self.locate_notebook("../../demos/ray_tracing.ipynb")
        replacements = {
            "n = 16": "n = 2",
            "d = 16": "d = 2",
            "px = 256": "px = 32",
            "parallel_x_blocks=1, parallel_y_blocks=1": "parallel_x_blocks=4, parallel_y_blocks=4",
        }
        self.run_notebook(notebook_path, replacements)


if __name__ == "__main__":
    unittest.main()
