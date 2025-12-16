import unittest
from __init__ import TB_TIMEOUT, BaseNotebookTest, run_notebook

class NotebookTests(BaseNotebookTest):

    def test_portfolio_allocation_demo_notebook(self):
        notebook_path, notebook_dir = self.locate_notebook('../../demos/portfolio/portfolio_allocation_demo.ipynb')
        value = [
            """params = [
    (4, 2**14),     
    (10, 2**15),   
    (20, 2**16),    
    (100, 2**17),   
    (500, 2**18),   
    (1000, 2**19)   
]""",
            "n_ports = [2**13, 2**14, 2**15]"
        ]
        new_value = [
            """params = [
    (4, 2**7),     
    (10, 2**8)]""",
            "n_ports = [2**7, 2**8]"
        ]
        run_notebook(notebook_path, notebook_dir, change_value=True, 
                     value=value, new_value=new_value, timeout=TB_TIMEOUT)

if __name__ == '__main__':
    unittest.main()
