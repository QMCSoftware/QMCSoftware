import unittest
from testbook import testbook

class NotebookTests(unittest.TestCase):

    @unittest.skip("Missing file path issue")
    @testbook('../../demos/digital_net_b2.ipynb', execute=True)
    def test_digital_net_b2_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
