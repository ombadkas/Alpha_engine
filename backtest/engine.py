"""
Portfolio construction and backtesting engine.

Strategy
--------
* Universe  : S&P 500 (current constituents, survivorship-bias caveat applies)
* Signal    : 3-factor composite (momentum + value + quality)
* Long leg  : equal-weight top quintile by composite score
* Short leg : equal-weight bottom quintile by composite score
* Rebalance : monthly (business month-end)
* Costs     : 5 bps one-way per trade (applied on portfolio turnover)

Implementation uses vectorbt's Portfolio.from_orders for transparent,
auditable position-level simulation.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

try:
    import vectorbt as vbt
    HAS_VBT = True
except ImportError:
    HAS_VBT = False

from config import BACKTEST_START, N_QUINTILES, FEES, REBAL_FREQ

log = logging.getLogger(__name__)


# ── Signal → weights ───────────────────────────────────────────────────────

def score_to_weights(
    scores: pd.Series,
    n_quintiles: int = N_QUINTILES,
) -> pd.Series:
    """
    Translate a cross-sectional score series into long/short target weights.

    Returns a Series with:
       +1/n_long   for top-quintile stocks (long)
        0          for middle quintiles
       -1/n_short  for bottom-quintile stocks (short)
    """
    valid = scores.dropna()
    if valid.empty:
        return pd.Series(0.0, index=scores.index)

    quantiles = pd.qcut(valid, n_quintiles, labels=False, duplicates="drop")

    top_mask    = quantiles == (n_quintiles - 1)
    bottom_mask = quantiles == 0

    n_long  = top_mask.sum()
    n_short = bottom_mask.sum()

    weights = pd.Series(0.0, index=scores.index)
    if n_long  > 0:
        weights[top_mask.index[top_mask]]    = +1.0 / n_long
    if n_short > 0:
        weights[bottom_mask.index[bottom_mask]] = -1.0 / n_short

    return weights


def build_weight_matrix(
    composite: pd.DataFrame,
    rebal_freq: str = REBAL_FREQ,
    n_quintiles: int = N_QUINTILES,
) -> pd.DataFrame:
    """
    Generate a (date × ticker) target-weight matrix.

    Weights are set on rebalancing dates; all other dates carry NaN
    (vectorbt will hold the prior position).
    """
    # Monthly rebalancing dates that exist in the index
    rebal_dates = composite.resample(rebal_freq).last().index
    rebal_dates = rebal_dates[rebal_dates.isin(composite.index)]

    weight_matrix = pd.DataFrame(np.nan, index=composite.index, columns=composite.columns)

    for dt in rebal_dates:
        scores = composite.loc[dt]
        weight_matrix.loc[dt] = score_to_weights(scores, n_quintiles)

    return weight_matrix


# ── Pandas portfolio simulation ────────────────────────────────────────────
# Used as primary engine (vectorbt wraps this for additional analytics).

def simulate_portfolio(
    close: pd.DataFrame,
    weights: pd.DataFrame,
    fees: float = FEES,
) -> dict:
    """
    Simulate a long-short portfolio from a target-weight matrix.

    Parameters
    ----------
    close   : (date × ticker) adjusted-close price matrix
    weights : (date × ticker) target weights; NaN means "hold"
    fees    : one-way transaction cost fraction (e.g. 0.0005 = 5 bps)

    Returns
    -------
    dict with keys:
        returns      : pd.Series of daily portfolio returns
        positions    : pd.DataFrame of daily holdings (target weights filled)
        turnover     : pd.Series of daily one-way turnover
    """
    # Forward-fill target weights so we know the position on every day
    positions = weights.ffill().fillna(0.0)

    # Align with close prices
    close, positions = close.align(positions, join="inner", axis=1)

    # Daily stock returns
    stock_rets = close.pct_change()

    # Portfolio return on day t = positions held at end of (t-1) × stock_ret(t)
    port_ret = (positions.shift(1) * stock_rets).sum(axis=1)

    # Turnover = sum of abs(new_weight - old_weight) on rebalancing days
    weight_change = positions.diff().abs()
    daily_turnover = weight_change.sum(axis=1) / 2  # divide by 2: double-counts legs

    # Transaction cost = turnover × fee (one-way)
    cost = daily_turnover * fees
    net_ret = port_ret - cost

    return {
        "returns"  : net_ret,
        "positions": positions,
        "turnover" : daily_turnover,
        "gross_ret": port_ret,
    }


# ── vectorbt wrapper ───────────────────────────────────────────────────────

def run_vbt_backtest(
    close: pd.DataFrame,
    weights: pd.DataFrame,
    fees: float = FEES,
    init_cash: float = 1_000_000,
) -> "vbt.Portfolio":
    """
    Run the backtest through vectorbt for rich analytics.

    Falls back gracefully to simulate_portfolio if vectorbt is unavailable.
    """
    if not HAS_VBT:
        log.warning("vectorbt not installed – using pure-pandas simulation.")
        return None

    # vectorbt from_orders expects a size matrix (target % of portfolio)
    # NaN rows = hold previous position (no order placed)
    order_size = weights.reindex(close.index)  # NaN between rebal dates → hold

    # Separate long and short allocations (vbt tracks them separately)
    long_size  = order_size.clip(lower=0)
    short_size = (-order_size).clip(lower=0)

    try:
        pf = vbt.Portfolio.from_orders(
            close=close,
            size=order_size,
            size_type="targetpercent",
            fees=fees,
            init_cash=init_cash,
            freq="D",
            group_by=False,   # per-asset tracking
            cash_sharing=True,
        )
        log.info("vectorbt portfolio constructed successfully.")
        return pf
    except Exception as exc:
        log.warning("vectorbt portfolio construction failed (%s); falling back.", exc)
        return None


# ── Main entry-point ───────────────────────────────────────────────────────

def run_backtest(
    close: pd.DataFrame,
    composite: pd.DataFrame,
    use_vbt: bool = True,
) -> dict:
    """
    Full backtest pipeline.

    Parameters
    ----------
    close     : (date × ticker) price matrix
    composite : (date × ticker) composite alpha score
    use_vbt   : attempt to use vectorbt (falls back to pandas on failure)

    Returns
    -------
    dict with:
        returns      : pd.Series daily net returns
        positions    : pd.DataFrame daily holdings
        turnover     : pd.Series daily turnover
        gross_ret    : pd.Series before-cost returns
        vbt_portfolio: vbt.Portfolio object (None if unavailable)
        weight_matrix: raw target-weight matrix
    """
    # Trim to backtest window
    close     = close.loc[BACKTEST_START:]
    composite = composite.loc[BACKTEST_START:]

    # Align tickers
    common_tickers = close.columns.intersection(composite.columns)
    close     = close[common_tickers]
    composite = composite[common_tickers]

    log.info(
        "Backtest: %s → %s  |  %d tickers",
        close.index[0].date(), close.index[-1].date(), len(common_tickers),
    )

    weight_matrix = build_weight_matrix(composite)
    result = simulate_portfolio(close, weight_matrix)

    vbt_pf = None
    if use_vbt and HAS_VBT:
        vbt_pf = run_vbt_backtest(close, weight_matrix)

    result["vbt_portfolio"] = vbt_pf
    result["weight_matrix"] = weight_matrix

    return result
