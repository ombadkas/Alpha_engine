"""Unit tests for factor computation modules."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from factors.momentum import raw_momentum, cross_sectional_rank, compute_momentum
from factors.value    import approximate_pb_series, compute_value, cross_sectional_zscore
from factors.quality  import compute_quality
from factors.combiner import zscore_signal, combine_factors, build_composite


# ── Fixtures ───────────────────────────────────────────────────────────────

def _make_prices(n_dates: int = 300, n_stocks: int = 20, seed: int = 42) -> pd.DataFrame:
    rng    = np.random.default_rng(seed)
    dates  = pd.date_range("2018-01-01", periods=n_dates, freq="B")
    tickers = [f"S{i:03d}" for i in range(n_stocks)]
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n_dates, n_stocks)), axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)


def _make_fundamentals(tickers: list[str], seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(tickers)
    return pd.DataFrame(
        {
            "priceToBook":    rng.uniform(0.5, 5.0, n),
            "returnOnEquity": rng.uniform(-0.1, 0.4, n),
            "trailingEps":    rng.uniform(0.5, 10.0, n),
            "bookValue":      rng.uniform(5.0, 50.0, n),
        },
        index=tickers,
    )


@pytest.fixture
def prices():
    return _make_prices()


@pytest.fixture
def fundamentals(prices):
    return _make_fundamentals(prices.columns.tolist())


# ── Momentum tests ─────────────────────────────────────────────────────────

class TestMomentum:
    def test_raw_momentum_shape(self, prices):
        mom = raw_momentum(prices, lookback=252, skip=21)
        assert mom.shape == prices.shape

    def test_raw_momentum_nan_warmup(self, prices):
        mom = raw_momentum(prices, lookback=252, skip=21)
        assert mom.iloc[:252].isna().all().all()

    def test_raw_momentum_positive_trending(self):
        """A stock that consistently trends up should have positive momentum."""
        dates = pd.date_range("2018-01-01", periods=300, freq="B")
        prices = pd.DataFrame(
            {"UP": np.linspace(100, 150, 300), "FLAT": np.full(300, 100.0)},
            index=dates,
        )
        mom = raw_momentum(prices, lookback=252, skip=21)
        last_row = mom.iloc[-1]
        assert last_row["UP"] > 0
        assert abs(last_row["FLAT"]) < 1e-10

    def test_cross_sectional_rank_bounds(self, prices):
        mom = raw_momentum(prices)
        ranked = cross_sectional_rank(mom)
        valid = ranked.dropna(how="all")
        assert valid.min().min() >= -0.5 - 1e-9
        assert valid.max().max() <=  0.5 + 1e-9

    def test_compute_momentum_no_inf(self, prices):
        result = compute_momentum(prices)
        assert not np.isinf(result.values).any()

    def test_skip_must_be_less_than_lookback(self, prices):
        with pytest.raises(ValueError):
            raw_momentum(prices, lookback=20, skip=21)


# ── Value tests ────────────────────────────────────────────────────────────

class TestValue:
    def test_value_shape(self, prices, fundamentals):
        val = compute_value(prices, fundamentals)
        assert val.shape == prices.shape

    def test_value_zscore_mean_approx_zero(self, prices, fundamentals):
        val = compute_value(prices, fundamentals).dropna(how="all")
        row_means = val.mean(axis=1).dropna()
        assert abs(row_means.mean()) < 0.5

    def test_cross_sectional_zscore_unit_std(self):
        rng  = np.random.default_rng(1)
        data = pd.DataFrame(rng.normal(10, 3, (50, 20)))
        z    = cross_sectional_zscore(data)
        row_stds = z.std(axis=1)
        assert ((row_stds - 1).abs() < 0.1).all()

    def test_missing_fundamentals_handled(self, prices):
        fund = pd.DataFrame(
            {"priceToBook": [np.nan] * len(prices.columns),
             "returnOnEquity": [np.nan] * len(prices.columns)},
            index=prices.columns,
        )
        result = compute_value(prices, fund)
        assert result.shape == prices.shape  # no exception raised


# ── Quality tests ──────────────────────────────────────────────────────────

class TestQuality:
    def test_quality_shape(self, prices, fundamentals):
        qual = compute_quality(prices, fundamentals)
        assert qual.shape == prices.shape

    def test_quality_no_inf(self, prices, fundamentals):
        qual = compute_quality(prices, fundamentals)
        assert not np.isinf(qual.values).any()


# ── Combiner tests ─────────────────────────────────────────────────────────

class TestCombiner:
    def test_zscore_clipping(self):
        """Values must be within [-clip, +clip]."""
        rng  = np.random.default_rng(2)
        data = pd.DataFrame(rng.normal(0, 5, (100, 30)))
        z    = zscore_signal(data, clip=3.0)
        assert z.min().min() >= -3.0 - 1e-9
        assert z.max().max() <=  3.0 + 1e-9

    def test_combine_equal_weights(self):
        rng    = np.random.default_rng(3)
        dates  = pd.date_range("2020-01-01", periods=100, freq="B")
        tickers = list("ABCDE")
        f1 = pd.DataFrame(rng.normal(0, 1, (100, 5)), index=dates, columns=tickers)
        f2 = pd.DataFrame(rng.normal(0, 1, (100, 5)), index=dates, columns=tickers)
        combined = combine_factors(f1, f2)
        assert combined.shape == f1.shape

    def test_combine_weights_must_sum_to_one(self):
        rng   = np.random.default_rng(4)
        f1    = pd.DataFrame(rng.normal(0, 1, (10, 5)))
        with pytest.raises(ValueError):
            combine_factors(f1, f1, weights=[0.6, 0.6])

    def test_build_composite_output_shape(self, prices, fundamentals):
        composite = build_composite(prices, fundamentals)
        # Composite is aligned to intersection of factor indices
        assert composite.shape[1] <= prices.shape[1]
        assert not composite.empty
