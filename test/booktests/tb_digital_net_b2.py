import unittest, pytest
from __init__ import BaseNotebookTest

class NotebookTests(BaseNotebookTest):

    def test_digital_net_b2_notebook(self):
        notebook_path, _ = self.locate_notebook('../../demos/digital_net_b2.ipynb')
        # Reduce sample sizes, trials and grid ranges to speed execution
        replacements = {
            's.gen_samples(2**25)': 's.gen_samples(2**12)',
            'trials = 100': 'trials = 4',
            'ms = arange(10,18)': 'ms = arange(10,12)',
            'plt_ei(s.gen_samples(2**6,warn=False)': 'plt_ei(s.gen_samples(2**3,warn=False)',
            'plt_ei(s.gen_samples(2**6,warn=False),ax[0,2],8,8)': 'plt_ei(s.gen_samples(2**3,warn=False),ax[0,2],2,2)',
            'plt_ei(s.gen_samples(2**6,warn=False),ax[1,2],16,4)': 'plt_ei(s.gen_samples(2**3,warn=False),ax[1,2],4,2)',
            's.gen_samples(n_min=1,n_max=16)': 's.gen_samples(n_min=1,n_max=4)',
            's.gen_samples(2**6,warn=False)': 's.gen_samples(2**3,warn=False)',
            's.gen_samples(2**4)': 's.gen_samples(2**2)'
        }
        self.run_notebook(notebook_path, replacements=replacements)

if __name__ == '__main__':
    unittest.main()
