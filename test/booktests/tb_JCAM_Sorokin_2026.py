import unittest
import os
from testbook import testbook
import sys

sys.path.insert(0, os.path.dirname(__file__))
from __init__ import TB_TIMEOUT, BaseNotebookTest, pip_install


@unittest.skip("Skipping NotebookTests class")
class NotebookTests(BaseNotebookTest):
    def setUp(self):
        super().setUp()  # Call parent setUp first to initialize timing attributes
        pip_install("tueplots")
        os.makedirs("outputs", exist_ok=True)

    @testbook(
        "../../demos/talk_paper_demos/JCAM_Sorokin_2026/JCAM_Sorokin_2026.ipynb",
        execute=False,
        timeout=TB_TIMEOUT,
    )
    def test_JCAM_Sorokin_2026_notebook(self, tb):
        pass


if __name__ == "__main__":
    unittest.main()
