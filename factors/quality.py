"""
Quality factor: Return on Equity (ROE) + accruals ratio.

Signal definition
-----------------
Primary signal: trailing ROE sourced from yfinance .info.
Secondary signal: operating-accruals ratio (approximated from price data as
a proxy for earnings quality — stocks with high accruals relative to assets
tend to underperform, Sloan 1996).

Composite quality =  0.5 * z(ROE) + 0.5 * z(-accruals_proxy)
                     ^^^^^^^^         ^^^^^^^^^^^^^^^^^^^^^^^^^^
                     higher is better  lower accruals = higher quality

All signals are cross-sectionally z-scored before combination.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from factors.value import cross_sectional_zscore


def roe_signal(
    close: pd.DataFrame,
    fundamentals: pd.DataFrame,
) -> pd.DataFrame:
    """
    Broadcast the point-in-time ROE scalar across the entire date range.

    In a production system this would be a PIT time-series updated quarterly.
    """
    roe = fundamentals.reindex(close.columns)["returnOnEquity"]

    # Create a constant time-series (same value every date)
    roe_df = pd.DataFrame(
        np.tile(roe.values, (len(close), 1)),
        index=close.index,
        columns=close.columns,
    )
    return roe_df


def accruals_proxy(
    close: pd.DataFrame,
    window: int = 252,
) -> pd.DataFrame:
    """
    Price-momentum dispersion as a crude accruals proxy.

    Uses the ratio of short-term to long-term momentum change as a noisy
    stand-in for accounting accruals when full balance-sheet data is absent.
    High values → potentially higher accruals → lower quality.
    """
    ret_short = close.pct_change(21)    # 1-month return
    ret_long  = close.pct_change(window) # 12-month return

    # dispersion ratio — high = earnings may be less sustainable
    proxy = ret_short.div(ret_long.abs().replace(0, np.nan))
    return proxy


def compute_quality(
    close: pd.DataFrame,
    fundamentals: pd.DataFrame,
    accruals_weight: float = 0.5,
) -> pd.DataFrame:
    """
    Full quality factor pipeline.

    Returns a (date × ticker) DataFrame of the composite quality signal.
    Higher values indicate higher quality.
    """
    z_roe = cross_sectional_zscore(roe_signal(close, fundamentals))

    proxy = accruals_proxy(close)
    z_acc = cross_sectional_zscore(-proxy)  # negate: lower accruals = better

    w_roe = 1.0 - accruals_weight
    composite = w_roe * z_roe + accruals_weight * z_acc

    # Final cross-sectional z-score of the composite
    return cross_sectional_zscore(composite)
