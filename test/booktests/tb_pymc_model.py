import unittest
from __init__ import BaseNotebookTest

class NotebookTests(BaseNotebookTest):

    def test_pymc_model_notebook(self):
        notebook_path, _ = self.locate_notebook("../../demos/pymc_model.ipynb")
        replacements = {
            "!unzip arma.json.zip": "!unzip -o arma.json.zip",
            "draws=2048, tune=1000": "draws=64, tune=64",
            "draws=1000, tune=1000": "draws=64, tune=64",
            "pm.sample()": "pm.sample(draws=64, tune=64)",
            "pm.sample(initvals=": "pm.sample(draws=64, tune=64, initvals=",
            "2**14": "2**7",
            "2**13": "2**7",
            "2**12": "2**7",
        }
        self.run_notebook(notebook_path, replacements)

if __name__ == '__main__':
    unittest.main()
