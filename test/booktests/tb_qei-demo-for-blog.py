import unittest
from testbook import testbook

class NotebookTests(unittest.TestCase):

    @testbook('../../demos/qei-demo-for-blog.ipynb', execute=True)
    def test_qei_demo_blog_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
