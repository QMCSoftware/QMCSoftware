import unittest
from __init__ import BaseNotebookTest

class NotebookTests(BaseNotebookTest):

    def test_ray_tracing_notebook(self):
        notebook_path, _ = self.locate_notebook('../../demos/ray_tracing.ipynb')
        replacements = {
            'n = 16': 'n = 8',
            'd = 16': 'd = 4',
            'px = 256': 'px = 64',
            'parallel_x_blocks=1, parallel_y_blocks=1': 'parallel_x_blocks=2, parallel_y_blocks=2'
        }
        self.run_notebook(notebook_path, replacements=replacements)

if __name__ == '__main__':
    unittest.main()
