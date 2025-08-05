import unittest
from testbook import testbook

class NotebookTests(unittest.TestCase):

    @unittest.skip("Excessive run time")
    @testbook('../../demos/ld_randomizations_and_higher_order_nets.ipynb', execute=True, timeout=1500)
    def test_ld_randomizations_and_higher_order_nets_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
