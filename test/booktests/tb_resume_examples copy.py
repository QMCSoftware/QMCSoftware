import unittest
from testbook import testbook
from __init__ import TB_TIMEOUT, BaseNotebookTest

class NotebookTests(BaseNotebookTest):

    @testbook('../../demos/demo_resume_data/resume_examples copy.ipynb', execute=True, timeout=TB_TIMEOUT)
    def test_resume_examples copy_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
