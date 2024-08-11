# -*- coding: utf-8 -*-
"""portfolio-allocation.ipynb

Original file is located at
    https://colab.research.google.com/drive/1XOLxjJ71fsTl3DUtr0zOitcLuvszKu37

MORE EFFICIENT PORTFOLIO ALLOCATION WITH QUASI-MONTE CARLO METHODS USING QMCPY
Larysa Matiukha and Sou-Cheng T. Choi

Illinois Institute of Technology

Modification date: 8/13/2023
Creation date: 8/1/2023
"""

import qmcpy as qp
import pandas as pd
import numpy as np
import yfinance as yf
import time
import timeit
import matplotlib.pyplot as plt
from qmcpy import *
import itertools


# To display the entire DataFrame without truncation
pd.set_option('display.max_columns', None)
# Setting the display precision for pandas DataFrame
pd.set_option('display.precision', 6)

class PortfolioAllocation:

    def __init__(self, start_date='2012-01-01', end_date='2023-08-04', split_date='2021-01-01',
                tickers=["AAPL", "AMZN", "CSCO", "IBM"],
                description=["Apple", "Amazon", "CISCO", "IBM"],
                num_ports=[2**14],
                seed=42, methods=["iid", "lattice", "sobol"]):
        self.start_date = start_date
        self.end_date = end_date
        self.split_date = split_date
        self.tickers = tickers
        self.description = description
        self.num_ports = num_ports
        self.seed = seed
        self.n = len(tickers)
        self.methods = methods
        self.output_filename = f"qmc_portfolio_{self.n}.csv"
        self.outputs = {}

    def _download_data(self):
        tickers = self.tickers
        description = self.description
        col_names = ['Ticker', 'Company', 'Date', 'Adj Close Price', 'Volume']
        df = pd.DataFrame(columns=col_names)

        for i, ticker in enumerate(tickers):
            company = description[i]
            data = yf.download(ticker, start=self.start_date, end=self.end_date)
            data['Ticker'] = ticker
            data['Company'] = company
            data.reset_index(inplace=True)
            data = data[['Ticker', 'Company', 'Date', 'Adj Close', 'Volume']]
            data.columns = col_names
            df = pd.concat([df, data], ignore_index=True)

        return df



    def _get_log_ret(self, df):
        stocks = df.pivot(index='Date', columns='Ticker', values='Adj Close Price')
        self.stocks_df = stocks
        log_ret = np.log(stocks / stocks.shift(1))
        return log_ret


    def gen_weights_lattice(self, n, num_ports):
        """ generating weights with randomized lattice points"""
        l = qp.Lattice(dimension=n, seed=self.seed, is_parallel=False)
        weights = l.gen_samples(num_ports)  # using lattice points instead of iid
        weights /= np.sum(weights, axis=1)[:, np.newaxis]
        return weights

    def gen_weights_sobol(self, n, num_ports):
        """ generating weights with randomized lattice points"""
        ld = qp.Sobol(n, seed=self.seed)  # define the generator
        weights = ld.gen_samples(num_ports)  # generate points
        weights /= np.sum(weights, axis=1)[:, np.newaxis]
        return weights
    def gen_weights_lattice_parallel(self, n, num_ports):
        """ generating weights with randomized lattice points"""
        l = qp.Lattice(dimension=n, seed=self.seed, order='mps', is_parallel=True)
        weights = l.gen_samples(num_ports)  # using lattice points instead of iid
        weights /= np.sum(weights, axis=1)[:, np.newaxis]
        return weights

    def gen_weights_iid(self, n, num_ports):
        """ generating weights with i.i.d. points"""
        np.random.seed(self.seed)
        # Generate all weights randomly from continuous uniform distribution in [0,1)
        weights = np.random.random((num_ports, n))
        weights /= np.sum(weights, axis=1)[:, np.newaxis]
        return weights


    def sharpe(self, weights, log_ret):
        """ Computing Sharpe Ratio
        Sharpe Ratio is given by: $\frac{R}{V}$, where $R = \sum_{i=1}^{d} R_iw_i$ stands for the excess expected return of a portfolio and $V = \sqrt{\sum_{i=1}^{d} \sum_{j=1}^{d} \sigma_{ij} w_i w_j}$ represents  its expected volatility with $d$ number of assets in a portfolio, weights $w_i$'s, and covariance of assets $i$ and $j$ $\sigma_{ij}$
        """
        np.random.seed(42)
        # Expected return： mean (log) return of each asset * weights
        ret_arr = np.sum((log_ret.mean().values * weights * 252), axis=1)

        # Expected volatility
        vol_arr = np.sqrt((weights @ (log_ret.cov().values * 252)) @ weights.T).diagonal()

        # Sharpe Ratio
        sharpe_arr = ret_arr / vol_arr

        if is_debug:
            log_ret.to_csv("log_returns.csv")
            pd.DataFrame(weights).to_csv("weights.csv")

        # Rounding values
        sharpe_arr = sharpe_arr
        ret_arr = ret_arr
        vol_arr = vol_arr
        all_weights = weights

        # max
        location_in_array = sharpe_arr.argmax()
        max_sr_ret = ret_arr[location_in_array]
        max_sr_vol = vol_arr[location_in_array]
        max_sharpe_ratio = max_sr_ret / max_sr_vol

        # mid
        medium_risk_tolerance = np.quantile(vol_arr, 2 / 3, axis=0)
        medium_risk_idx = np.where(vol_arr < medium_risk_tolerance)
        location_in_array = sharpe_arr[medium_risk_idx].argmax()
        medium_risk_sr_ret = ret_arr[medium_risk_idx][location_in_array]
        medium_risk_sr_vol = vol_arr[medium_risk_idx][location_in_array]
        medium_risk_max_sharpe_ratio = medium_risk_sr_ret / medium_risk_sr_vol

        # low
        low_risk_tolerance = np.quantile(vol_arr, 1 / 3, axis=0)
        low_risk_idx = np.where(vol_arr < low_risk_tolerance)
        location_in_array = sharpe_arr[low_risk_idx].argmax()
        low_risk_sr_ret = ret_arr[low_risk_idx][location_in_array]
        low_risk_sr_vol = vol_arr[low_risk_idx][location_in_array]
        low_risk_max_sharpe_ratio = low_risk_sr_ret / low_risk_sr_vol

        output_dict = {"number of tickers": len(weights[0]),
                "number of portfolios": len(weights),
                "low": all_weights[low_risk_idx][location_in_array].tolist(),
                "medium": all_weights[medium_risk_idx][location_in_array].tolist(),
                "high": all_weights[location_in_array].tolist(),
                "low risk Sharpe": low_risk_max_sharpe_ratio,
                "medium risk Sharpe": medium_risk_max_sharpe_ratio,
                "high risk Sharpe": max_sharpe_ratio}

        return output_dict

    def download_data(self):
        df = self._download_data()

        # Split data into in-sample and out-of-sample
        self.df_in_sample = df[df['Date'] < self.split_date].copy(deep=True)
        self.df_out_sample = df[df['Date'] >= self.split_date].copy(deep=True)

        # log return for the 4 stocks
        self.lr = self._get_log_ret(self.df_in_sample)

    def _optimize_one(self, method, n, num_port):
        if method == "lattice":
            w = self.gen_weights_lattice(n, num_port)
        if method == "lattice_parallel":
            w = self.gen_weights_lattice_parallel(n, num_port)
        elif method == "iid":
            w = self.gen_weights_iid(n, num_port)
        elif method == "sobol":
            w = self.gen_weights_sobol(n, num_port)

        sr = self.sharpe(w, self.lr)
        data = {
            'Sampling Method': method,
            'Number of Assets': [self.n],
            'Number of Portfolios': [num_port],
            'Low-risk max. Sharpe': [sr['low risk Sharpe']],
            'Medium-risk max. Sharpe': [sr['medium risk Sharpe']],
            'High-risk max. Sharpe': [sr['high risk Sharpe']],
        }
        df = pd.DataFrame(data)
        self.outputs[method] = {}
        self.outputs[method]["Sharpe ratios"] = df.copy(deep=True)
        self.outputs[method]["low"] = sr['low']
        self.outputs[method]["medium"] = sr['medium']
        self.outputs[method]["high"] = sr['high']

        return df

    def optimize(self):
        df = None
        for num_port in self.num_ports:
            for method in self.methods:
                df_row = self._optimize_one(method, self.n, num_port)
                df = df_row if df is None else pd.concat([df, df_row])
        df.reset_index(inplace=True, drop=True)
        df.to_csv(self.output_filename, float_format='%.6f')
        return df

    def backtest(self, principal=10000, risk_level="high", start_date=None, end_date=None):

        # Extract out-of-sample adjusted close prices into a DataFrame
        stocks_df_out = self.df_out_sample.pivot(index='Date', columns='Ticker', values='Adj Close Price')

        # Calculate normalized returns for out-of-sample data
        stocks_df = self.df_out_sample.copy(deep=True)
        stocks_df = stocks_df.pivot(index='Date', columns='Ticker', values='Adj Close Price')

        for method in self.methods:
            stocks_df[f'{method} Total Pos'] = 0
            for ticker, allocation in zip(self.tickers, self.outputs[method][risk_level]):
                stock_df = stocks_df[[ticker]].copy(deep=True)
                stock_df.columns = [f'Adj Close Price']
                stock_df['Norm Return'] = stock_df[f'Adj Close Price'] / stock_df.iloc[0][f'Adj Close Price']
                stock_df['Allocation'] = stock_df[f'Norm Return'] * allocation
                stock_df['Position'] = stock_df[f'Allocation'] * principal
                stocks_df[f'{method} Total Pos'] = stocks_df[f'{method} Total Pos'] + stock_df['Position']

        # plt.style.use('Solarize_Light2')
        qmc_vs_mc_columns = [f'{m} Total Pos' for m in self.methods]
        stocks_df[qmc_vs_mc_columns].plot(
            title=f'Out-Sample Backtest: {risk_level.capitalize()}-Risk Daily Portfolio Values Using Weights \nCorresponding to Maximum Sharpe Ratio from {self.num_ports[0]} Sampling Points',
            logy=True)
        plt.ylabel('Cumulative Portfolio Value ($)')
        legend = [s.capitalize() if s.lower() != "iid" else s for s in self.methods]
        sharpe = [self.outputs[m]['Sharpe ratios'][f'{risk_level.capitalize()}-risk max. Sharpe'] for m in self.methods]
        sharpe = [round(t.values[0], 4) for t in sharpe]
        plt.legend([l + ", Sharpe ratio = " + str(s) for l, s in zip(legend, sharpe)])
        # plt.show()
        plt.savefig(f'qmc_vs_mc_backtest_{risk_level}_risk.png')

    def backtest0(self, principal=10000, risk_level="high", start_date=None, end_date=None):
        """ principal: starting investment value
        """
        if start_date is None:
            start_date = self.split_date
        if end_date is None:
            end_date = self.end_date
        stocks_df = self.stocks_df

        stocks_df_out = self._get_log_ret(self.df_out_sample).cumsum()
        stocks_df_out = np.exp(stocks_df_out) * principal

        # normalize return
        for method in self.methods:
            stocks_df[f'{method} Total Pos'] = 0
            for ticker, allocation in zip(self.tickers, self.outputs[method][risk_level]):
                stock_df = stocks_df[[ticker]].copy(deep=True)
                stock_df.columns = [f'Adj Close Price']
                stock_df['Norm Return'] = stock_df[f'Adj Close Price'] / stock_df.iloc[0][f'Adj Close Price']
                stock_df['Allocation'] = stock_df[f'Norm Return'] * allocation
                stock_df['Position'] = stock_df[f'Allocation'] * principal
                stocks_df[f'{method} Total Pos'] = stocks_df[f'{method} Total Pos'] + stock_df['Position']

        #plt.style.use('Solarize_Light2')
        qmc_vs_mc_columns = [f'{m} Total Pos' for m in self.methods]
        stocks_df[qmc_vs_mc_columns].plot(title=f'Backtest: {risk_level.capitalize()}-Risk Daily Portfolio Values Using Weights \nCorresponding to Maximum Sharpe Ratio and Starting Principal ${principal}', logy=True)
        plt.ylabel('Cumulative Portfolio Value ($)')
        legend = [s.capitalize() if s.lower() != "iid" else s for s in self.methods]
        sharpe = [self.outputs[m]['Sharpe ratios'][f'{risk_level.capitalize()}-risk max. Sharpe'] for m in self.methods]
        sharpe = [round(t.values[0], 4) for t in sharpe]
        plt.legend([l + ", Sharpe ratio = " + str(s) for l, s in zip(legend, sharpe)])
        #plt.show()
        plt.savefig(f'qmc_vs_mc_backtest_{risk_level}_risk.png')



if __name__ == "__main__":
    is_debug = False
    """ Testing
    We now test the two approaches by computing Sharpe ratios at different risk levels employing the weights obtained from each method.
    
    As we can see from the tables below, the values of Sharpe ratio do not significantly change based on the number of portfolios. We might want to look furher into this: will results change more significantly if we triple the number of portfolios for the tickers we worked with, or what happens if we use much greater number of tickers to begin with.
 
    """
    is_test_small_eg = False
    is_test_mc_qmc = False
    is_measure_time = False
    is_backtest = False
    is_plot_simplex = False
    is_out_sample_backtest = False
    is_count_highest_sharpe_ratios = True

    if is_test_small_eg:
        tickers = ["AAPL", "AMZN"]
        description = ["Apple", "Amazon"]
        methods = ["iid", "lattice", "sobol"]
        ns = range(2, 3)
        num_ports = [2 ** n for n in ns]
        ao = PortfolioAllocation(tickers=tickers, description=description, methods=methods, num_ports=num_ports,
                                 start_date="2013-01-01", end_date="2013-01-09")
        ao.download_data()
        df = ao.optimize()
        print(df)

    if is_test_mc_qmc:
        tickers = ["AAPL", "AMZN", "CSCO", "IBM"]
        description = ["Apple", "Amazon", "CISCO", "IBM"]
        methods = ["iid", "lattice", "sobol"]
        ns = range(10, 14)
        num_ports = [2**n for n in ns]
        ao = PortfolioAllocation(methods=methods, num_ports=num_ports,)
        ao.download_data()
        df = ao.optimize()
        print(df)

        tickers = ["AAPL", "AMZN", "CSCO", "IBM", "TSLA", "META", "ABNB", "UPS", "NFLX", "MRNA"]
        description = ["Apple", "Amazon", "CISCO", "IBM", "Tesla", "Meta", "Airbnb", "UPS", "Netflix", "Moderna"]
        ao = PortfolioAllocation(tickers=tickers, description=description, methods=methods, num_ports=num_ports)
        ao.download_data()
        df = ao.optimize()
        print(df)

        tickers = ["AAPL", "AMZN", "CSCO", "IBM", "TSLA", "META", "ABNB", "UPS", "NFLX", "MRNA", "^IXIC", "T", "GE", "FMC",
                    "AMC", "JPM", "DIS", "WBA", "GOOGL", "BA"]
        description = ["Apple", "Amazon", "CISCO", "IBM", "Tesla", "Meta", "Airbnb", "UPS", "Netflix", "Moderna", "NASDAQ",
                        "AT&T", "General Electric", "FMC", "AMC", "JPMorgan", "Disney", "WBA", "Google", "Boeing"]
        ao = PortfolioAllocation(tickers=tickers, description=description, methods=methods, num_ports=num_ports)
        ao.download_data()
        df = ao.optimize()
        print(df)

    """ measuring runtime
    ### real time
    """
    if is_measure_time:
        ao = PortfolioAllocation()
        # Define configurations
        configs = [
            (4, 6000),
            (10, 20000),
            (20, 30000),
            (100, 150000),
         #   (500, 200000),
         #   (1000, 500000)
        ]

        results = []
        for time_method, time_method_name in [(time.time, 'Clock time'), (time.process_time, 'CPU time')]:
            for tickers, portfolios in configs:
                for method, method_name in [(ao.gen_weights_iid, 'iid', ),
                                        (ao.gen_weights_lattice, 'lattice'),
                                        (ao.gen_weights_lattice_parallel, 'lattice_parallel'),
                                        (ao.gen_weights_lattice_parallel, 'sobol')]:

                    start_time = time_method()
                    _ = method(tickers, portfolios)  # Assuming you don't need the tr values
                    elapsed_time = time_method() - start_time

                    #print(f"takes to run: {elapsed_time} for {tickers} tickers, {portfolios} portfolios ({method_name})")
                    results.append({
                        'Method': method_name,
                        'Assets': tickers,
                        'Portfolios': portfolios,
                        f'{time_method_name}': elapsed_time
                    })

            # Convert results to a pandas dataframe
            time_df = pd.DataFrame(results)
            print(time_df)
            time_df.to_csv(f"runtime_{time_method_name}.csv")

        """Here is the table of real runtime comparison for Monte Carlo and Quasi Monte Carlo Methods
        """
        # Change ntickers (assets in portfolios) and fix number of sampling points (portolios)
        seed = 42
        trials = 3
        ntickers_values = [50, 100, 200, 400, 800, 1600]  # Example values for ntickers
        n = 13  # Fixed value for n as given
        time_output = {}
        for time_method, time_method_name in [(time.time, 'Clock time'), (time.process_time, 'CPU time')]:
            plt.figure()  # Create a new figure for each iteration
            time_output[time_method_name] = {}
            for method, method_name, marker in \
                    [(ao.gen_weights_iid, 'IID', 'o'),
                     (ao.gen_weights_lattice, 'Lattice', '+'),
                     (ao.gen_weights_lattice_parallel, 'Lattice Multi-threaded', 's'),
                     (ao.gen_weights_sobol, 'Sobol', '>')]:

                ts = []
                for ntickers in ntickers_values:
                    start_time = time_method()
                    for _ in range(0, trials):
                        method(ntickers, 2 ** n)
                    end_time = time_method() - start_time
                    ts.append(end_time / trials)

                # store the time outputs
                time_output[time_method_name] = {**time_output[time_method_name], method_name: ts}

                #  Plot time for each method
                plt.plot(ntickers_values, ts, color='indigo',
                         marker=marker, linestyle='none', label=method_name,
                         markerfacecolor='none', markeredgewidth=1.5)

            plt.title(f"Average Runtime of {trials} Trials for Each Method with {2 ** n} Portfolios")
            plt.xlabel("Number of Assets")
            plt.ylabel(f"{time_method_name} (seconds)")
            plt.yscale('log')
            plt.xscale('log')
            plt.legend()
            plt.savefig(f'runtime_{time_method_name}_ntickers.png')

            # compute lattice speed up by multithreading
            x = time_output[time_method_name]['Lattice']
            y = time_output[time_method_name]["Lattice Multi-threaded"]
            speedup = [xx / yy for xx, yy in zip(x, y)]
            df = pd.DataFrame({'ntickers': ntickers_values, 'speedup': speedup})

            # Create a bar chart for speed up
            plt.figure()
            ax = df.plot.bar(x='ntickers', y='speedup', rot=0,legend=None)
            # Add labels and title
            ax.set_xlabel('Number of Tickers')
            ax.set_ylabel('Speed-Up Factor')
            ax.set_title(f'Speed-Up Factor of Multi-Threaded Lattice Points Generation of \n {2 ** n} Points Using {time_method_name} vs. Number of Tickers')
            plt.savefig(f'speedup_lattice_{time_method_name}_ntickers.png')

        # ------------------------------------------------------------------------------------------------------
        # fix ntickers (assets in portfolios) and change number of sampling points (portolios)
        # ------------------------------------------------------------------------------------------------------

        ns = range(5, 15)
        seed = 42
        ntickers = 1000
        trials = 3
        time_output = {}
        for time_method, time_method_name in [(time.time, 'Clock time'), (time.process_time, 'CPU time')]:
            plt.figure()  # Create a new figure
            time_output[time_method_name] = {}
            for method, method_name, marker in \
                    [(ao.gen_weights_iid, 'IID',  'o'),
                     (ao.gen_weights_lattice, 'Lattice',  '+'),
                     (ao.gen_weights_lattice_parallel, 'Lattice Multi-threaded', 's'),
                     (ao.gen_weights_sobol, 'Sobol', '>')]:
                ts = []
                for n in ns:
                    start_time = time_method()
                    for _ in range(0, trials):
                        method(ntickers, 2**n)
                    end_time = time_method() - start_time
                    ts.append(end_time/trials)

                # store the time outputs
                time_output[time_method_name] = {**time_output[time_method_name], method_name: ts}

                # Plot time for each method
                plt.plot([2**n for n in ns], ts, color='indigo',
                         marker=marker, linestyle='none', label=method_name,
                         markerfacecolor='none', markeredgewidth=1.5)

            plt.title(f"Average Runtime of {trials} Trials for Each Method with {ntickers} Assets")
            plt.xlabel("Number of Portfolios")
            plt.ylabel(f"{time_method_name} (seconds)")
            plt.yscale('log')
            plt.xscale('log')
            plt.legend()
            plt.savefig(f'runtime_{time_method_name}.png')

            # compute lattice speed up by multithreading
            x = time_output[time_method_name]['Lattice']
            y = time_output[time_method_name]["Lattice Multi-threaded"]
            speedup = [xx / yy for xx, yy in zip(x, y)]
            df = pd.DataFrame({'portfolios': [2**n for n in ns], 'speedup': speedup})

            # Create a bar chart for speed up
            plt.figure()
            ax = df.plot.bar(x='portfolios', y='speedup', rot=0, legend=None)
            # Add labels and title
            ax.set_xlabel('Number of Portfolios')
            ax.set_ylabel('Speed-Up Factor')
            ax.set_title(f'Speed-Up Factor of Multi-Threaded Lattice Points Generation for \n{ntickers} Assets Using {time_method_name} vs. Number of Samples')
            plt.savefig(f'speedup_lattice_{time_method_name}_samples.png')


    if is_backtest:
        num_ports = [2**14]
        methods = ["iid", "lattice", "sobol"]
        ao = PortfolioAllocation(methods=methods, num_ports=num_ports)
        ao.download_data()
        df = ao.optimize()
        for risk_level in ["low", "medium", "high"]:
            ao.backtest(principal=10000, risk_level=risk_level)

    if is_plot_simplex:
        import numpy as np
        import matplotlib.pyplot as plt

        # Define the combinations of indices for the subplots
        combinations = [(0, 1), (1, 2), (0, 2)]
        xlabels = ["$x_1$", "$x_2$", "$x_1$"]
        ylabels = ["$x_2$", "$x_3$", "$x_3$"]

        # Points on a simplex
        n = 2 ** 8  # number of sampling points
        d = 3  # dimension of the simplex
        seed = 42
        ao = PortfolioAllocation(seed=seed)
        for method, method_name in [(ao.gen_weights_iid, 'IID',),
                                    (ao.gen_weights_lattice, 'Lattice'),
                                    (ao.gen_weights_lattice_parallel, 'Lattice Mutlti-threaded'),
                                    (ao.gen_weights_sobol, 'Sobol')]:
            w = method(d, n)

            # Create a main figure with three subplots
            fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # 3 subplots in a row

            # Fill in each subplot
            for ax, (idx1, idx2), xlabel, ylabel in zip(axs, combinations, xlabels, ylabels):
                ax.scatter(w[:, idx1], w[:, idx2], marker='.')
                ax.axis([0, 1, 0, 1])
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)

            title_text = (
                f"{n = } {method_name} Sampling Points in the Standard Simplex with $d=3$\n"
                r"$\left\{ x \in \mathbb{R}^d: x = \left(x_1, \ldots x_{d}\right), x_{i} \geq 0, \sum_{i=1}^d x_{i} = 1 \right\}$"
            )
            fig.suptitle(title_text, y=1)

            # Use tight_layout to adjust spacing
            plt.tight_layout()
            plt.savefig(f"{method_name}Simplex_seed{seed}.png")


        def gen_points_iid(d, n, seed):
            np.random.seed(seed)
            # Generate all weights randomly from continuous uniform distribution in [0,1)
            points = np.random.random((n, d))
            return points

        def gen_points_lattice(d, n, seed):
            l = qp.Lattice(dimension=d, seed=seed, order='mps')
            points = l.gen_samples(n)  # using lattice points instead of iid
            return points

        def gen_points_sobol(d, n, seed):
            l = qp.Sobol(dimension=d, seed=seed)
            points = l.gen_samples(n)  # using lattice points instead of iid
            return points
        """
        for method, method_name in [(gen_points_iid, 'IID'),
                                    (gen_points_lattice, 'Lattice')]:
            w = method(d, n, seed)
            plt.figure(figsize=(6, 6))  # Set a square figure size
            plt.scatter(w[:, 0], w[:, 1], marker='.')
            plt.axis([0, 1, 0, 1])
            title_text = (
                f"{n = } {method_name} Sampling Points in the Unit Cube with $d=3$\n"
                r"$\left\{ x \in \mathbb{R}^d: x = \left(x_1, \ldots x_{d}\right), 0 \leq x_{i} \leq 1\right\}$"
            )
            plt.title(title_text, pad=10)
            plt.xlabel("$x_1$")
            plt.ylabel("$x_2$")
            # Use tight_layout to adjust spacing
            plt.tight_layout()
            plt.savefig(f"{method_name}Cube_seed{seed}.png")
        """

        # Iterate over the methods
        for method, method_name in [(gen_points_iid, 'IID'), (gen_points_lattice, 'Lattice'), (gen_points_sobol,'Sobol')]:
            w = method(d, n, seed)

            # Create a main figure with three subplots
            fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # 3 subplots in a row

            # Fill in each subplot
            for ax, (idx1, idx2), xlabel, ylabel in zip(axs, combinations, xlabels, ylabels):
                ax.scatter(w[:, idx1], w[:, idx2], marker='.')
                ax.axis([0, 1, 0, 1])
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)

            title_text = (
                f"{n = } {method_name} Sampling Points in the Unit Cube with $d=3$\n"
                r"$\left\{ x \in \mathbb{R}^d: x = \left(x_1, \ldots x_{d}\right), 0 \leq x_{i} \leq 1\right\}$"
            )
            fig.suptitle(title_text, y=1)

            # Use tight_layout to adjust spacing
            plt.tight_layout()
            plt.savefig(f"{method_name}Cube_seed{seed}.png")

    if is_out_sample_backtest:
        # Initialize the portfolio allocation object
        portfolio = PortfolioAllocation(num_ports=[2**14], start_date='2018-01-01', end_date='2023-08-04', split_date='2023-01-01')

        # Download and preprocess data
        portfolio.download_data()

        # Optimize the portfolio based on in-sample data
        portfolio.optimize()

        # Backtest the optimized portfolio on out-of-sample data
        for risk_level in ["low", "medium", "high"]:
            portfolio.backtest(risk_level=risk_level)

    if is_count_highest_sharpe_ratios:
        trials = 1000

        portfolio = PortfolioAllocation(num_ports=[2**8],
                                        start_date='2018-01-01',
                                        end_date='2023-08-04',
                                        split_date='2023-01-01',
                                        seed=None)

        portfolio.download_data()

        nmethods = len(portfolio.methods)
        rank1_counts = {'Low-risk max. Sharpe': [0]*nmethods,
                       'Medium-risk max. Sharpe': [0]*nmethods,
                       'High-risk max. Sharpe': [0]*nmethods}
        for i in range(0, trials):
            # Optimize the portfolio based on in-sample data
            df = portfolio.optimize()
            #print(df)

            # Calculate ranks for the last three columns
            ranks = df.iloc[:, -3:].apply(lambda col: col.rank(method='min', ascending=False))

            # Find the index of rank 1 for each column
            for col in ranks.columns:
                rank_1_index = ranks[col][ranks[col] == 1].index[0]
                rank1_counts[col][rank_1_index] =rank1_counts[col][rank_1_index]+1

        rank1_counts_df = pd.DataFrame(rank1_counts, index=rank1_counts.keys())
        rank1_counts_df.columns = portfolio.methods
        print(rank1_counts_df)