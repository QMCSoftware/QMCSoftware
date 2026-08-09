import unittest
from testbook import testbook
from __init__ import TB_TIMEOUT, BaseNotebookTest, pip_install


class NotebookTests(BaseNotebookTest):

    def setUp(self):
        super().setUp()
        # Install required packages
        pip_install("matplotlib")

    @testbook(
        "../../demos/lattice_random_generator.ipynb", execute=True, timeout=TB_TIMEOUT
    )
    def test_lattice_random_generator_notebook(self, tb):
        pass


if __name__ == "__main__":
    unittest.main()
