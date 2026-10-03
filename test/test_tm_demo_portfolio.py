"""Unit tests for the portfolio demo's backtest computation.

Covers demos/portfolio/backtest_util.py: point-in-time universe construction,
periodic/event-driven rebalancing, and Sharpe-ratio computation. Fast,
isolated regression guards for bugs found and fixed in this module: phantom
capital on a late listing, phantom liquidation of a temporarily absent
ticker, and a unit mismatch between an annualized risk-free rate and daily
returns, each previously caught only informally or not at all, as distinct
from the slow, whole-notebook check in
test/booktests/tb_portfolio_allocation_demo.py.

Classes:
    TestListingChangeDates: universe-membership-change detection; no randomization.
    TestStopLossDates: drawdown/floor trigger detection; no randomization.
    TestComputePortfolioValueReps: the backtest itself, from plain buy-and-
        hold up to combined rebalancing and stop-loss; all inputs are small,
        deterministic toy price series, no randomization.
    TestComputeAllPortfolios: forwarding across samplers/risk levels.
    TestSharpeReps: Sharpe-ratio computation, with and without a risk-free
        rate; no randomization.

Example:
    python3 -m pytest test/test_tm_demo_portfolio.py
"""
from unittest import TestCase

import numpy as np
import pandas as pd
import pytest

bu = pytest.importorskip("demos.portfolio.backtest_util")


def _stock_df(prices, dates):
    """One ticker's DataFrame with 'Adj Close Price' and 'Norm Return', as
    setup_stock_dfs would build it."""
    df = pd.DataFrame({"Adj Close Price": list(prices)}, index=pd.DatetimeIndex(dates))
    df["Norm Return"] = df["Adj Close Price"] / df["Adj Close Price"].iloc[0]
    return df


class TestListingChangeDates(TestCase):
    """No replication: every case is a single deterministic call."""

    def test_same_window_no_changes(self):
        """A shared, uninterrupted window has exactly one change: day 1."""
        dates = pd.bdate_range("2020-01-01", periods=5)
        a = _stock_df([10.0] * 5, dates)
        b = _stock_df([20.0] * 5, dates)
        assert bu.listing_change_dates([a, b]) == [dates[0]]

    def test_late_listing_detected(self):
        """A ticker listing partway through registers its own listing and delisting."""
        dates = pd.bdate_range("2020-01-01", periods=5)
        a = _stock_df([10.0] * 5, dates)
        b = _stock_df([20.0, 21.0], dates[2:4])
        assert bu.listing_change_dates([a, b]) == [dates[0], dates[2], dates[4]]

    def test_temporary_halt_detected(self):
        """A gap in one ticker's own index (not a shared weekend) is a halt/resume pair."""
        dates = pd.bdate_range("2020-01-01", periods=7)
        a = _stock_df([10.0] * 7, dates)
        halted_dates = dates[[0, 1, 2, 5, 6]]
        b = _stock_df([20.0] * 5, halted_dates)
        assert bu.listing_change_dates([a, b]) == [dates[0], dates[3], dates[5]]


class TestStopLossDates(TestCase):
    """No replication: every case is a single deterministic call."""

    def test_no_trigger_without_conditions(self):
        """Neither condition given: never triggers, regardless of the price path."""
        dates = pd.bdate_range("2020-01-01", periods=5)
        a = _stock_df([10.0, 9.0, 8.0, 7.0, 6.0], dates)
        assert bu.stop_loss_dates([a]) == {}

    def test_drop_pct_fires_on_first_breach(self):
        """Fires the first day price falls below (1 - drop_pct) times the running peak."""
        dates = pd.bdate_range("2020-01-01", periods=5)
        a = _stock_df([10.0, 11.0, 9.0, 5.0, 5.0], dates)
        # running peak after day 2 (index 1) is 11.0; a 20% drawdown threshold is 8.8.
        # day index 2 (9.0) is still above 8.8; day index 3 (5.0) is the first breach.
        triggers = bu.stop_loss_dates([a], stop_loss_drop_pct=0.2)
        assert triggers == {0: dates[3]}

    def test_price_floor_fires(self):
        """Fires the first day price falls below an absolute floor."""
        dates = pd.bdate_range("2020-01-01", periods=4)
        a = _stock_df([10.0, 8.0, 6.0, 4.0], dates)
        triggers = bu.stop_loss_dates([a], stop_loss_price_floor=7.0)
        assert triggers == {0: dates[2]}

    def test_earliest_of_both_conditions_wins(self):
        """When both conditions are given, the earliest-firing one determines the date."""
        dates = pd.bdate_range("2020-01-01", periods=5)
        a = _stock_df([10.0, 10.0, 7.0, 4.0, 4.0], dates)
        # floor=8.0 breaches first, at index 2 (7.0 < 8.0); drop_pct=0.5 from a peak of
        # 10 doesn't breach (5.0 threshold) until index 3 (4.0 < 5.0), one day later.
        triggers = bu.stop_loss_dates([a], stop_loss_drop_pct=0.5, stop_loss_price_floor=8.0)
        assert triggers == {0: dates[2]}

    def test_per_ticker_floor_exempts_none(self):
        """A sequence of floors applies independently per ticker; None exempts one."""
        dates = pd.bdate_range("2020-01-01", periods=3)
        a = _stock_df([10.0, 5.0, 1.0], dates)
        b = _stock_df([10.0, 5.0, 1.0], dates)
        triggers = bu.stop_loss_dates([a, b], stop_loss_price_floor=[6.0, None])
        assert triggers == {0: dates[1]}


class TestComputePortfolioValueReps(TestCase):
    """No replication: every case uses a single deterministic weights row."""

    def test_buy_and_hold_matches_old_formula(self):
        """rebalance_freq=None reproduces the pre-fix formula exactly, including its
        phantom-capital bug: a late listing contributes zero the whole time instead
        of being added once listed (preserved for backward compatibility)."""
        toy = pd.DataFrame({
            "Ticker": ["A", "A", "A", "B"],
            "Date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-02"]),
            "Adj Close Price": [10.0, 10.0, 11.0, 10.0],
        })
        dfs = bu.setup_stock_dfs(toy, ["A", "B"])
        v = bu.compute_portfolio_value_reps(
            [dfs["A"], dfs["B"]], np.array([[0.5, 0.5]]), 100
        ).iloc[:, 0]
        assert v.tolist() == [50.0, 100.0, 55.00000000000001]

    def test_rebalancing_fixes_phantom_capital(self):
        """BLOCKER 13: with rebalance_freq set, a late listing is added at its fair
        value once listed, instead of leaving its allocation uninvested until then."""
        toy = pd.DataFrame({
            "Ticker": ["A", "A", "A", "B"],
            "Date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-02"]),
            "Adj Close Price": [10.0, 10.0, 11.0, 10.0],
        })
        dfs = bu.setup_stock_dfs(toy, ["A", "B"])
        v = bu.compute_portfolio_value_reps(
            [dfs["A"], dfs["B"]], np.array([[0.5, 0.5]]), 100, rebalance_freq="D"
        ).iloc[:, 0]
        assert v.tolist() == [100.0, 100.0, 100.0]

    def test_absent_ticker_freezes_not_liquidates(self):
        """Extends BLOCKER 13: a ticker that goes permanently absent (bankruptcy or a
        data feed simply stopping) has its dollar balance frozen, not redistributed
        to the other tickers, and no rebalance_freq=None-style cliff appears either."""
        dates = pd.bdate_range("2020-01-01", periods=120)
        x = _stock_df(100 * (1.0003 ** np.arange(len(dates))), dates)
        y_dates = dates[:100]
        y = _stock_df(50 * (0.998 ** np.arange(100)), y_dates)
        weights = np.array([[0.5, 0.5]])

        v_none = bu.compute_portfolio_value_reps([x, y], weights, 10_000, rebalance_freq=None).iloc[:, 0]
        v_fixed = bu.compute_portfolio_value_reps([x, y], weights, 10_000, rebalance_freq="QS").iloc[:, 0]
        li, li_next = y_dates[-1], dates[100]

        jump_none = v_none.loc[li_next] - v_none.loc[li]
        jump_fixed = v_fixed.loc[li_next] - v_fixed.loc[li]
        assert jump_none < -3_000, "no rebalancing should write Y's value off as an instant cliff"
        assert abs(jump_fixed) < 10, "freezing Y's balance should leave no cliff at all"

    def test_universe_change_reacts_same_day(self):
        """A late listing is picked up the day it lists, not the next scheduled date,
        when rebalance_on_universe_change=True; the default (False) waits."""
        dates = pd.bdate_range("2020-01-01", periods=10)
        a = _stock_df([10.0] * 10, dates)
        # B jumps 20% the day after listing, so an allocation that includes B shows up
        # as a value difference the next day; flat prices throughout would leave both
        # modes at exactly the $100 principal with nothing to distinguish them.
        b = _stock_df([20.0, 24.0, 24.0, 24.0, 24.0], dates[3:8])
        weights = np.array([[0.5, 0.5]])

        v_default = bu.compute_portfolio_value_reps(
            [a, b], weights, 100, rebalance_freq="QS"
        ).iloc[:, 0]
        v_event = bu.compute_portfolio_value_reps(
            [a, b], weights, 100, rebalance_freq="QS", rebalance_on_universe_change=True
        ).iloc[:, 0]

        # B lists on dates[3]; with no quarter boundary in this short window, the
        # default never re-triggers on B's listing at all, so B stays uninvested.
        assert v_default.loc[dates[3]] == 100.0
        # The event-driven mode rebalances the instant B lists, so B's half appears.
        assert v_event.loc[dates[3]] == 100.0
        assert v_event.loc[dates[4]] != v_default.loc[dates[4]]

    def test_stop_loss_never_reenters(self):
        """A triggered stop-loss sells at that day's price, folds proceeds into the
        survivors, and the sold ticker's later price action no longer matters."""
        dates = pd.bdate_range("2020-01-01", periods=6)
        a = _stock_df([100.0] * 6, dates)
        # b falls 50% on day index 2, then recovers fully by the end; a stop-loss
        # should still have sold at the index-2 low and never bought back in.
        b = _stock_df([100.0, 100.0, 50.0, 100.0, 100.0, 100.0], dates)
        weights = np.array([[0.5, 0.5]])

        v = bu.compute_portfolio_value_reps(
            [a, b], weights, 100, rebalance_freq="D", stop_loss_drop_pct=0.1
        ).iloc[:, 0]
        # By the end, b's recovery should not be reflected: everything should be in a.
        assert v.iloc[-1] == pytest.approx(v.loc[dates[2]])

    def test_combined_floor_and_drop_pct(self):
        """stop_loss_price_floor and stop_loss_drop_pct can both be supplied; whichever
        condition fires first for a ticker determines its sale date."""
        dates = pd.bdate_range("2020-01-01", periods=4)
        a = _stock_df([100.0] * 4, dates)
        b = _stock_df([100.0, 100.0, 40.0, 40.0], dates)
        weights = np.array([[0.5, 0.5]])
        exit_dates = bu.stop_loss_dates([a, b], stop_loss_drop_pct=0.9, stop_loss_price_floor=50.0)
        assert exit_dates == {1: dates[2]}
        v = bu.compute_portfolio_value_reps(
            [a, b], weights, 100, rebalance_freq="D", stop_loss_price_floor=[None, 50.0]
        ).iloc[:, 0]
        assert v.iloc[-1] == pytest.approx(v.loc[dates[2]])

    def test_unequal_stock_count_raises(self):
        """A mismatched weights/stock_dfs width is a configuration error, not silently ignored."""
        dates = pd.bdate_range("2020-01-01", periods=2)
        a = _stock_df([10.0, 10.0], dates)
        with pytest.raises(ValueError):
            bu.compute_portfolio_value_reps([a], np.array([[0.5, 0.5]]), 100)


class TestComputeAllPortfolios(TestCase):
    def test_forwards_sampler_and_risk_level(self):
        """One DataFrame per (sampler, risk level), each a real backtest."""
        dates = pd.bdate_range("2020-01-01", periods=4)
        a = _stock_df([10.0] * 4, dates)
        b = _stock_df([20.0] * 4, dates)
        sr_dict = {
            "sobol": {"low": np.array([[0.5, 0.5]]), "high": np.array([[0.8, 0.2]])},
            "iid": {"low": np.array([[0.3, 0.7]]), "high": np.array([[0.9, 0.1]])},
        }
        out = bu.compute_all_portfolios([a, b], sr_dict, ["low", "high"], 100)
        assert set(out) == {"sobol", "iid"}
        for sampler in out:
            assert set(out[sampler]) == {"low", "high"}
            for risk in out[sampler]:
                assert out[sampler][risk].iloc[0, 0] == pytest.approx(100.0)


class TestSharpeReps(TestCase):
    """No replication (R=1), one portfolio (P=1), one ticker (D=1): the three
    risk tiers all select that same single portfolio, so low/medium/high
    Sharpe come out identical every time; only the Sharpe formula itself,
    not the risk-tier selection, is under test here."""

    def test_no_rf_uses_raw_return(self):
        """Without a risk-free rate, excess_ret is log_ret itself."""
        dates = pd.bdate_range("2020-01-01", periods=2)
        log_ret = pd.DataFrame({"A": [0.02, 0.00]}, index=dates)
        weights = np.array([[[1.0]]])
        sr = bu.sharpe_reps(weights, log_ret)
        assert sr["low risk Sharpe"] == pytest.approx(11.225)
        assert sr["medium risk Sharpe"] == pytest.approx(11.225)
        assert sr["high risk Sharpe"] == pytest.approx(11.225)

    def test_rf_uses_daily_equivalent(self):
        """Regression test: log_rf is annualized and must be divided by 252
        before subtracting from log_ret's own daily returns. For this data,
        the pre-fix formula (subtracting the annualized rate directly) gave
        a Sharpe ratio of -45.349 instead of the correct 11.0."""
        dates = pd.bdate_range("2020-01-01", periods=2)
        log_ret = pd.DataFrame({"A": [0.02, 0.00]}, index=dates)
        log_rf = pd.Series([0.0504, 0.0504], index=dates)
        weights = np.array([[[1.0]]])
        sr = bu.sharpe_reps(weights, log_ret, log_rf)
        assert sr["low risk Sharpe"] == pytest.approx(11.0)
        assert sr["medium risk Sharpe"] == pytest.approx(11.0)
        assert sr["high risk Sharpe"] == pytest.approx(11.0)

    def test_se_nan_with_one_replication(self):
        """An SE needs at least two replications; R=1 (every other test here) gives NaN."""
        dates = pd.bdate_range("2020-01-01", periods=2)
        log_ret = pd.DataFrame({"A": [0.02, 0.00]}, index=dates)
        weights = np.array([[[1.0]]])
        sr = bu.sharpe_reps(weights, log_ret)
        assert np.isnan(sr["medium risk Sharpe SE"])

    def test_se_across_replications(self):
        """Two tickers with equal variance but different mean return, one
        replication all-in on each: the two replications' Sharpe ratios are
        11.225 and 22.450, so the SE of their mean is their half-difference,
        5.612, independent of risk tier since P=1 forces the same portfolio
        into every tier."""
        dates = pd.bdate_range("2020-01-01", periods=2)
        log_ret = pd.DataFrame({"A": [0.02, 0.00], "B": [0.01, 0.03]}, index=dates)
        weights = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])
        sr = bu.sharpe_reps(weights, log_ret)
        assert sr["medium risk Sharpe"] == pytest.approx(16.837)
        assert sr["medium risk Sharpe SE"] == pytest.approx(5.612)
