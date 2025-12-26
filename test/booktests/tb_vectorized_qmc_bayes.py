import unittest
from testbook import testbook
from __init__ import TB_TIMEOUT, BaseNotebookTest

class NotebookTests(BaseNotebookTest):

    def test_vectorized_qmc_bayes_notebook(self):
        # open the notebook with testbook, replace occurrences in all cells then execute
        with testbook('../../demos/vectorized_qmc_bayes.ipynb', execute=False, timeout=TB_TIMEOUT) as tb:
            for cell in tb.nb.get('cells', []):
                src = cell.get('source', '')
                new = src.replace('max_iter=1024', 'max_iter=10') \
                     .replace('abs_tol=.005', 'abs_tol=.1, n_limit=2**8')
                if new != src:
                    cell['source'] = new
            tb.execute()

if __name__ == '__main__':
    unittest.main()
