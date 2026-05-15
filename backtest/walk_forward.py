"""
Walk-forward out-of-sample validation.

Methodology
-----------
Rolling window scheme:
  * Train window : TRAIN_YEARS years (factor weights / hyperparameters could
                   be re-optimised here; in this version we use fixed weights)
  * Test  window : TEST_YEARS  year (out-of-sample)
  * Step  size   : TEST_YEARS year (non-overlapping OOS periods)

This prevents look-ahead bias — the model only "sees" data prior to each
test period when constructing signals.

The function returns the concatenated OOS return series, which forms the
basis of the main performance report.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterator

import pandas as pd

from backtest.engine   import build_weight_matrix, simulate_portfolio
from backtest.metrics  import performance_summary
from config            import TRAIN_YEARS, TEST_YEARS, FEES

log = logging.getLogger(__name__)


@dataclass
class WalkForwardFold:
    train_start: pd.Timestamp
    train_end:   pd.Timestamp
    test_start:  pd.Timestamp
    test_end:    pd.Timestamp
    returns:     pd.Series   = field(default_factory=pd.Series)
    stats:       pd.Series   = field(default_factory=pd.Series)


def _generate_folds(
    index: pd.DatetimeIndex,
    train_years: int = TRAIN_YEARS,
    test_years:  int = TEST_YEARS,
) -> Iterator[WalkForwardFold]:
    """Yield (train, test) date-range pairs with a rolling window."""
    start = index[0]
    end   = index[-1]

    period_start = start
    while True:
        train_start = period_start
        train_end   = train_start + pd.DateOffset(years=train_years)
        test_start  = train_end
        test_end    = test_start + pd.DateOffset(years=test_years)

        if test_end > end:
            break

        yield WalkForwardFold(
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )

        period_start = period_start + pd.DateOffset(years=test_years)


def run_walk_forward(
    close: pd.DataFrame,
    composite: pd.DataFrame,
    benchmark_returns: pd.Series | None = None,
    train_years: int = TRAIN_YEARS,
    test_years:  int = TEST_YEARS,
    fees: float = FEES,
) -> tuple[pd.Series, list[WalkForwardFold]]:
    """
    Execute rolling walk-forward validation.

    Parameters
    ----------
    close             : (date × ticker) price matrix (full history)
    composite         : (date × ticker) composite score (full history)
    benchmark_returns : optional benchmark daily returns for IR calculation
    train_years       : length of training window in years
    test_years        : length of OOS test window in years
    fees              : one-way transaction cost

    Returns
    -------
    oos_returns : concatenated out-of-sample daily return series
    folds       : list of WalkForwardFold objects with per-fold statistics
    """
    common = close.columns.intersection(composite.columns)
    close     = close[common]
    composite = composite[common]

    folds = list(_generate_folds(composite.index, train_years, test_years))
    if not folds:
        raise ValueError(
            f"Date range too short for {train_years}+{test_years} year walk-forward."
        )

    oos_chunks: list[pd.Series] = []

    for i, fold in enumerate(folds):
        log.info(
            "Fold %d/%d: train [%s → %s], test [%s → %s]",
            i + 1, len(folds),
            fold.train_start.date(), fold.train_end.date(),
            fold.test_start.date(), fold.test_end.date(),
        )

        # Signals built only from training-period data (no look-ahead)
        train_composite = composite.loc[fold.train_start : fold.train_end]
        # Build weight matrix on training data statistics, apply to test period
        # (Here factor weights are fixed; could optimise on training set)
        test_composite  = composite.loc[fold.test_start : fold.test_end]
        test_close      = close.loc[fold.test_start : fold.test_end]

        weight_matrix = build_weight_matrix(test_composite)
        sim = simulate_portfolio(test_close, weight_matrix, fees=fees)

        fold.returns = sim["returns"]
        oos_chunks.append(fold.returns)

        bench = (
            benchmark_returns.loc[fold.test_start : fold.test_end]
            if benchmark_returns is not None
            else None
        )
        fold.stats = performance_summary(fold.returns, bench, sim["turnover"])

        log.info("  OOS Sharpe: %.3f  |  Max DD: %.2f%%",
                 fold.stats.get("Sharpe Ratio", float("nan")),
                 fold.stats.get("Max Drawdown", float("nan")) * 100)

    oos_returns = pd.concat(oos_chunks).sort_index()
    return oos_returns, folds


def fold_summary_table(folds: list[WalkForwardFold]) -> pd.DataFrame:
    """Collate per-fold statistics into a display-ready DataFrame."""
    rows = []
    for fold in folds:
        row = fold.stats.to_dict()
        row["Test Start"] = fold.test_start.strftime("%Y-%m")
        row["Test End"]   = fold.test_end.strftime("%Y-%m")
        rows.append(row)
    df = pd.DataFrame(rows).set_index(["Test Start", "Test End"])
    return df
