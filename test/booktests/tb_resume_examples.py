import unittest
from __init__ import BaseNotebookTest

class NotebookTests(BaseNotebookTest):

    def test_resume_examples_notebook(self):
        notebook_path, notebook_dir = self.locate_notebook(
            "../../demos/demo_resume_data/resume_examples.ipynb"
        )
        symlinks_to_fix = [
            "resume_util.py",
        ]
        self.fix_symlinks(notebook_dir, symlinks_to_fix)
        self.run_notebook(notebook_path)

if __name__ == '__main__':
    unittest.main()
