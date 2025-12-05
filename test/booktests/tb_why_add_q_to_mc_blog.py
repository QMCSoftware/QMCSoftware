import unittest
from testbook import testbook
from __init__ import TB_TIMEOUT, BaseNotebookTest

class NotebookTests(BaseNotebookTest):
    @testbook('../../demos/talk_paper_demos/why_add_q_to_mc_blog/why_add_q_to_mc_blog.ipynb', execute=True, timeout=TB_TIMEOUT)
    def test_why_add_q_to_mc_blog_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
