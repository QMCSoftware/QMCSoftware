import unittest
from testbook import testbook
from __init__ import TB_TIMEOUT, BaseNotebookTest

@pytest.mark.slow 
class NotebookTests(BaseNotebookTest):

    @testbook('../../demos/qei-demo-for-blog.ipynb', execute=True, timeout=TB_TIMEOUT)
    def test_qei_demo_for_blog_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
