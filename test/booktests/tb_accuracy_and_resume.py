import unittest
from testbook import testbook
from __init__ import TB_TIMEOUT, BaseNotebookTest

class NotebookTests(BaseNotebookTest):

    @testbook('../../demos/demo_resume_data/accuracy_and_resume.ipynb', execute=True, timeout=TB_TIMEOUT)
    def test_accuracy_and_resume_notebook(self, tb):
        notebook_path, notebook_dir = self.locate_notebook(
            "../../demos/demo_resume_data/accuracy_and_resume.ipynb"
        )
        symlinks_to_fix = [
            "resume_util.py",
        ]
        self.fix_gbm_symlinks(notebook_dir, symlinks_to_fix)
        self.run_notebook(notebook_path)


if __name__ == '__main__':
    unittest.main()
