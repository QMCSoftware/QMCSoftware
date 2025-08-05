import unittest
from testbook import testbook

class NotebookTests(unittest.TestCase):

    @unittest.skip("Missing LookBackOption class")
    @testbook('../../demos/PricingAsianOptions.ipynb', execute=True)
    def test_pricing_asian_options_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
