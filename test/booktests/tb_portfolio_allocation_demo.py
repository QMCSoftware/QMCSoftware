import unittest
from __init__ import BaseNotebookTest

class NotebookTests(BaseNotebookTest):

    def test_portfolio_allocation_demo_notebook(self):
        notebook_path, _ = self.locate_notebook('../../demos/portfolio/portfolio_allocation_demo.ipynb')
        
        # Define variables for readability 
        old_tick1 = '"CSCO", "IBM", "TSLA", "META", "ABNB", "UPS", "NFLX", "MRNA"'
        old_desc1 = '"CISCO", "IBM", "Tesla", "Meta", "Airbnb", "UPS", "Netflix", "Moderna"'
        old_tick2 = '"IBM","TSLA","META","ABNB","UPS","NFLX","MRNA","^IXIC", "T","GE","FMC","AMC","JPM","DIS","CVX","GOOGL","BA"'
        old_desc2 = '"IBM","Tesla","Meta","Airbnb","UPS","Netflix","Moderna","NASDAQ","AT&T","General Electric","FMC","AMC","JPMorgan","Disney","Chevron","Google","Boeing"'
        
        replacements = {
            "(10, 2**15),":"",
            "(20, 2**16),":"",
            "(20, 2**16),":"",   
            "(100, 2**17),":"",
            "(500, 2**18),":"",   
            "(1000, 2**19)":"",
            "n_ports = [2**13, 2**14, 2**15]": "n_ports = [2**7, 2**8]",
            "start_date = '2014-01-01'": "start_date = '2025-06-01'",
            "dimensions = [5, 10, 20, 50, 100, 200, 500, 1000]": "dimensions = [5, 10]",
            "num_ports = 2**14": "num_ports = 8",
            f'tickers1 = ["AAPL", "AMZN", {old_tick1}]': 'tickers1 = ["AAPL", "AMZN"]',
            f'description1 = ["Apple", "Amazon", {old_desc1}]': 'description1 = ["Apple", "Amazon"]',
            f'tickers2 = ["AAPL", "AMZN", "CSCO",{old_tick2}]': 'tickers2 = ["AAPL", "AMZN", "CSCO"]',
            f'description2 = ["Apple", "Amazon", "CISCO", {old_desc2}]': 'description2 = ["Apple", "Amazon", "CISCO"]',
        }
        
        self.run_notebook(notebook_path, replacements)