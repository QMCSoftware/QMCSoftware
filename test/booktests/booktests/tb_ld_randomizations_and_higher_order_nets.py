import unittest
from testbook import testbook

class NotebookTests(unittest.TestCase):

    @testbook('../../demos/ld_randomizations_and_higher_order_nets.ipynb', execute=True)
    def test_ld_randomizations_and_higher_order_nets_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
