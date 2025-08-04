import unittest
from testbook import testbook

class NotebookTests(unittest.TestCase):

    @testbook('../../demos/saving_qmc_state.ipynb', execute=True)
    def test_saving_qmc_state_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
