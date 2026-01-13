import unittest, pytest
import os
import shutil
import subprocess
from __init__ import BaseNotebookTest


@pytest.mark.slow
class NotebookTests(BaseNotebookTest):

    notebook_dir = "../../demos/talk_paper_demos/JOSS2025/"
    notebook_path = f"{notebook_dir}joss2025.ipynb"

    def setUp(self):
        super().setUp()  # Call parent setUp first to initialize timing attributes
        subprocess.run(
            ["pip", "install", "-q", "seaborn>=0.13.0", "tueplots"], check=False
        )
        # Create the JOSS2025.outputs directory that the notebook expects
        self.output_dir = os.path.join(self.notebook_dir, "JOSS2025.outputs")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
            self._created_output_dir = True
        else:
            self._created_output_dir = False

    def tearDown(self):
        # Clean up the created directory if we created it
        if hasattr(self, "_created_output_dir") and self._created_output_dir:
            if os.path.exists(self.output_dir):
                shutil.rmtree(self.output_dir)
        super().tearDown()

    def test_joss2025_notebook(self):
        replacements = {
            "trials = 100": "trials = 2",
            "assert os.path.isdir(OUTDIR)": "",
            "fig.savefig": "#fig.savefig",
            "np.save": "#np.save",
        }

        self.run_notebook(self.notebook_path, replacements)


if __name__ == "__main__":
    unittest.main()
