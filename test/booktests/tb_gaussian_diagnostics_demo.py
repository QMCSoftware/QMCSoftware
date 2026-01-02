import unittest, pytest
import subprocess
import os
from __init__ import BaseNotebookTest

@pytest.mark.slow
class NotebookTests(BaseNotebookTest):

    def setUp(self):
        super().setUp()
        # Install required packages
        subprocess.run(['pip', 'install', '-q', 'matplotlib', 'scipy', 'seaborn==0.8'], check=False)
        # Create outputs directory if needed
        os.makedirs('outputs_nb', exist_ok=True)

    def test_gaussian_diagnostics_demo_notebook(self):
        notebook_path, _ = self.locate_notebook('../../demos/gaussian_diagnostics/gaussian_diagnostics_demo.ipynb')
        # Speed up heavy computations and plotting in the demo
        replacements = {
            'nRep = 20': 'nRep = 4',
            'nRep = 5': 'nRep = 3',
            'npts = 2 ** 6': 'npts = 2 ** 5',
            'npts = 2 ** 10': 'npts = 2 ** 8',
            'lnthetarange = np.arange(-2, 2.2, 0.2)': 'lnthetarange = np.arange(-2, 2.2, 0.8)',
            "lnorderrange = np.arange(-1, 1.1, 0.1)": "lnorderrange = np.arange(-1, 1.1, 0.4)",
            'xtol=1e-3': 'xtol=1e-2',
            'N1 = int(2 ** np.floor(16 / dim))': 'N1 = int(2 ** np.floor(12 / dim))'
        }
        self.run_notebook(notebook_path, replacements=replacements)

if __name__ == '__main__':
    unittest.main()