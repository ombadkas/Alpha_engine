"""Unit tests for backtest engine and metrics."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from backtest.engine      import score_to_weights, build_weight_matrix, simulate_portfolio
from backtest.metrics     import (
    annualised_return, annualised_volatility, sharpe_ratio,
    max_drawdown, calmar_ratio, information_ratio,
    rolling_sharpe, monthly_returns, performance_summary,
)
from backtest.walk_forward import run_walk_forward


# ── Fixtures ───────────────────────────────────────────────────────────────

def _make_prices(n_dates: int = 800, n_stocks: int = 50, seed: int = 7) -> pd.DataFrame:
    rng     = np.random.default_rng(seed)
    dates   = pd.date_range("2015-01-01", periods=n_dates, freq="B")
    tickers = [f"S{i:03d}" for i in range(n_stocks)]
    prices  = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, (n_dates, n_stocks)), axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)


def _make_composite(prices: pd.DataFrame, seed: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(0, 1, prices.shape),
        index=prices.index,
        columns=prices.columns,
    )


@pytest.fixture
def prices():
    return _make_prices()


@pytest.fixture
def composite(prices):
    return _make_composite(prices)


# ── Engine tests ───────────────────────────────────────────────────────────

class TestScoreToWeights:
    def test_weights_sum_to_zero(self):
        """Long-short portfolio: gross weights should sum near zero."""
        scores = pd.Series(np.random.default_rng(1).normal(0, 1, 50))
        weights = score_to_weights(scores, n_quintiles=5)
        assert abs(weights.sum()) < 1e-9

    def test_long_weights_positive(self):
        scores  = pd.Series(range(100), dtype=float)
        weights = score_to_weights(scores, n_quintiles=5)
        assert (weights[weights > 0] > 0).all()

    def test_short_weights_negative(self):
        scores  = pd.Series(range(100), dtype=float)
        weights = score_to_weights(scores, n_quintiles=5)
        assert (weights[weights < 0] < 0).all()

    def test_long_short_equal_gross_exposure(self):
        scores  = pd.Series(range(100), dtype=float)
        weights = score_to_weights(scores, n_quintiles=5)
        long_gross  = weights[weights > 0].sum()
        short_gross = weights[weights < 0].abs().sum()
        assert abs(long_gross - short_gross) < 1e-9

    def test_empty_scores(self):
        scores = pd.Series(dtype=float)
        weights = score_to_weights(scores)
        assert weights.empty

    def test_all_nan_returns_zeros(self):
        scores = pd.Series([np.nan] * 10)
        weights = score_to_weights(scores)
        assert (weights == 0).all()


class TestBuildWeightMatrix:
    def test_shape(self, prices, composite):
        wm = build_weight_matrix(composite)
        assert wm.shape == composite.shape

    def test_rebal_dates_not_all_nan(self, prices, composite):
        wm = build_weight_matrix(composite)
        non_nan_rows = wm.notna().any(axis=1).sum()
        assert non_nan_rows > 0

    def test_off_rebal_rows_are_nan(self, prices, composite):
        wm = build_weight_matrix(composite)
        # Most rows between rebalancing dates should be all-NaN
        nan_rows = wm.isna().all(axis=1).sum()
        assert nan_rows > len(wm) * 0.8  # >80% of rows are non-rebalancing


class TestSimulatePortfolio:
    def test_returns_series_length(self, prices, composite):
        wm     = build_weight_matrix(composite)
        result = simulate_portfolio(prices, wm)
        assert len(result["returns"]) == len(prices)

    def test_turnover_non_negative(self, prices, composite):
        wm     = build_weight_matrix(composite)
        result = simulate_portfolio(prices, wm)
        assert (result["turnover"] >= 0).all()

    def test_net_less_than_gross(self, prices, composite):
        wm     = build_weight_matrix(composite)
        result = simulate_portfolio(prices, wm, fees=0.001)
        net   = result["returns"]
        gross = result["gross_ret"]
        # Net should be ≤ gross on most rebalancing days
        diff = gross - net
        assert (diff >= -1e-12).all()

    def test_positions_sum_approx_zero(self, prices, composite):
        """L/S portfolio: long + short weights should roughly cancel."""
        wm     = build_weight_matrix(composite)
        result = simulate_portfolio(prices, wm)
        pos_sums = result["positions"].sum(axis=1)
        # Mean net exposure should be close to zero
        assert abs(pos_sums.mean()) < 0.2


# ── Metrics tests ──────────────────────────────────────────────────────────

class TestMetrics:
    def _ret(self, mu=0.0005, sigma=0.01, n=500, seed=0):
        rng = np.random.default_rng(seed)
        return pd.Series(rng.normal(mu, sigma, n))

    def test_annualised_return_sign(self):
        pos = self._ret(mu=0.001)
        neg = self._ret(mu=-0.001)
        assert annualised_return(pos) > 0
        assert annualised_return(neg) < 0

    def test_sharpe_positive_for_positive_drift(self):
        returns = self._ret(mu=0.001, sigma=0.005)
        assert sharpe_ratio(returns) > 0

    def test_max_drawdown_negative(self):
        returns = self._ret()
        assert max_drawdown(returns) <= 0

    def test_max_drawdown_zero_for_always_positive(self):
        returns = pd.Series([0.001] * 100)
        assert max_drawdown(returns) == 0.0

    def test_calmar_positive_for_positive_drift(self):
        returns = self._ret(mu=0.001, sigma=0.005)
        assert calmar_ratio(returns) > 0

    def test_information_ratio_positive_alpha(self):
        strat = self._ret(mu=0.001, sigma=0.01)
        bench = self._ret(mu=0.0005, sigma=0.01, seed=1)
        assert information_ratio(strat, bench) > 0

    def test_rolling_sharpe_length(self):
        returns = self._ret(n=500)
        rs = rolling_sharpe(returns, window=252)
        assert len(rs) == len(returns)
        assert rs.iloc[:251].isna().all()   # rolling(252) first fires at position 251

    def test_monthly_returns_columns(self):
        returns = self._ret(n=252 * 3, mu=0.0003)
        returns.index = pd.date_range("2020-01-01", periods=len(returns), freq="B")
        pivot = monthly_returns(returns)
        assert set(pivot.columns).issubset(
            {"Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"}
        )

    def test_performance_summary_returns_series(self):
        returns = self._ret()
        returns.index = pd.date_range("2020-01-01", periods=len(returns), freq="B")
        bench   = self._ret(seed=5)
        bench.index = returns.index
        summary = performance_summary(returns, bench)
        assert isinstance(summary, pd.Series)
        assert "Sharpe Ratio" in summary.index
        assert "Max Drawdown" in summary.index


# ── Walk-forward tests ─────────────────────────────────────────────────────

class TestWalkForward:
    def test_oos_returns_cover_expected_period(self, prices, composite):
        oos, folds = run_walk_forward(prices, composite, train_years=2, test_years=1)
        assert len(folds) >= 1
        assert len(oos) > 0

    def test_folds_non_overlapping(self, prices, composite):
        _, folds = run_walk_forward(prices, composite, train_years=2, test_years=1)
        for i in range(1, len(folds)):
            assert folds[i].test_start >= folds[i - 1].test_end

    def test_each_fold_has_stats(self, prices, composite):
        _, folds = run_walk_forward(prices, composite, train_years=2, test_years=1)
        for fold in folds:
            assert "Sharpe Ratio" in fold.stats.index
