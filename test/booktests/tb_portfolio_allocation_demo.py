import unittest, pytest
from __init__ import BaseNotebookTest
import tb_data_portfolio_allocation

@pytest.mark.slow
class NotebookTests(BaseNotebookTest):

    def test_data_portfolio_allocation(self):
        """Run the data portfolio allocation notebook test"""
        # Force run even if skip decorator is present
        test_instance = tb_data_portfolio_allocation.NotebookTests()
        test_instance.setUp()
        # Call the actual test method directly, bypassing skip
        notebook_path, _ = test_instance.locate_notebook('../../demos/portfolio/data_portfolio_allocation.ipynb')
        test_instance.run_notebook(notebook_path)

    def test_portfolio_allocation_demo_notebook(self):
        notebook_path, _ = self.locate_notebook(
            "../../demos/portfolio/portfolio_allocation_demo.ipynb"
        )

        replacements = {
            "n_ports = [2**13, 2**14, 2**15]": "n_ports = [2**7, 2**8]",
            "dimensions = [5, 10, 20, 50, 100, 200, 500, 1000]": "dimensions = [5, 10]",
            "range(8, 18)": "range(8, 11)",
            "fixed_dimension = 50": "fixed_dimension = 10",
            "num_ports = 2**14": "num_ports = 2**5",
            'tickers1, description1 = load_assets(f"data/df10_{start_date}_to_{end_date}.csv")': 'tickers1, description1 = ["AAPL", "ABNB"], ["Apple", "Airbnb"]',
            # The backtest cell hardcodes n_tickers instead of deriving it from
            # len(tickers1); keep it in sync with the tickers1 shrink above.
            "n_tickers = 10": "n_tickers = 2",
        }

        self.run_notebook(notebook_path, replacements)

if __name__ == "__main__":
    unittest.main()
