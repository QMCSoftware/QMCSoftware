import os
import unittest
from testbook import testbook
import shutil, tempfile, glob

def create_notebook_test(notebook_path):
    """Factory function to create test methods for notebooks"""
    # compute the directory where the notebook lives
    nb_dir = os.path.dirname(notebook_path)
    
    # pass that into kernel_kwargs so the kernel's cwd is the demos folder
    @testbook(
        notebook_path,
        execute=True,
        kernel_kwargs={'cwd': nb_dir}
    )
    def test_method(self, tb):
        # nothing to do here—TestBook already executed the notebook
        pass

    return test_method

class FastNotebookTests(unittest.TestCase):
    """Test class for fast-running demo notebooks (< 30s)"""
    # no need for demos_path here now

    def tearDown(self):
        """Clean up any output files generated in the test directory"""
        test_dir = os.getcwd()
        # Patterns of files to remove
        patterns = ['*.txt*', '*.csv', '*.out', '*.part', '*.txt']
        for pattern in patterns:
            for fname in glob.glob(os.path.join(test_dir, pattern)):
                try:
                    temp_dir = os.path.join(tempfile.gettempdir(), "notebook_test_outputs")
                    os.makedirs(temp_dir, exist_ok=True)
                    shutil.move(fname, os.path.join(temp_dir, os.path.basename(fname)))
                except OSError:
                    pass

# list of fast notebooks
fast_notebooks = [
    'quickstart.ipynb',
    'qmcpy_intro.ipynb',
    'integration_examples.ipynb',
    'lebesgue_integration.ipynb',
    'plot_proj_function.ipynb',
    'some_true_measures.ipynb',
]

# dynamically attach tests
for nb in fast_notebooks:
    path = f'../../demos/{nb}'
    name = f"test_{nb.replace('.ipynb','').replace('-', '_')}_notebook"
    setattr(FastNotebookTests, name, create_notebook_test(path))

if __name__ == '__main__':
    unittest.main()