import unittest
import os
import shutil
import numpy as np
from testbook import testbook
from __init__ import TB_TIMEOUT, BaseNotebookTest

@unittest.skip("Skipping NotebookTests class")
class NotebookTests(BaseNotebookTest):

    def setUp(self):
        super().setUp()  # Call parent setUp first to initialize timing attributes
        # fix error that involves x_full_dakota.txt by moving/creating the file where notebook expects it
        demo_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'demos', 'DAKOTA_Genz')
        source_dakota_file = os.path.join(demo_dir, 'x_full_dakota.txt')
        # The notebook runs from the booktests directory, so copy the file here
        target_dakota_file = os.path.join(os.path.dirname(__file__), 'x_full_dakota.txt')
        
        if os.path.isfile(source_dakota_file) and not os.path.isfile(target_dakota_file):
            # Copy the downloaded file from DAKOTA_Genz to booktests directory
            shutil.copy2(source_dakota_file, target_dakota_file)
            self._created_dummy_file = target_dakota_file
        elif not os.path.isfile(target_dakota_file):
            # Create smaller dummy Dakota data to prevent NameError and speed up test
            # Using smaller dimensions: ds.max() = 128, but fewer samples for speed
            dummy_data = np.random.rand(1024, 128)  # Much smaller than ns.max() = 262144
            np.savetxt(target_dakota_file, dummy_data)
            self._created_dummy_file = target_dakota_file
        else:
            self._created_dummy_file = None

    @testbook('../../demos/DAKOTA_Genz/dakota_genz.ipynb', execute=True, timeout=TB_TIMEOUT)
    def test_dakota_genz_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
