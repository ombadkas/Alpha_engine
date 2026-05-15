"""
Low-volatility factor (low-vol anomaly).

Empirical observation (Black, Jensen & Scholes 1972; Haugen & Heins 1975):
stocks with lower realised volatility tend to produce higher risk-adjusted
returns — and, in many regimes, higher raw returns — than high-vol stocks.

Signal definition
-----------------
For each stock on date t, compute the annualised realised volatility over
the trailing `window` trading days.  Negate it so that low-vol stocks get
a HIGH score (they are preferred in the long leg).  Cross-sectionally
z-score so the signal is on the same scale as the other factors.

This signal is computed entirely from price data — no fundamental data
required — making it robust to the snapshot-data limitation of yfinance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from factors.value import cross_sectional_zscore   # reuse z-score utility

TRADING_DAYS = 252


def trailing_realised_vol(close: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """
    Annualised trailing realised volatility over `window` trading days.

    Parameters
    ----------
    close  : (date × ticker) adjusted-close price matrix
    window : lookback in trading days (~63 = 3 months)

    Returns
    -------
    (date × ticker) DataFrame of annualised vol.  First `window` rows are NaN.
    """
    daily_ret = close.pct_change(fill_method=None)
    roll_std  = daily_ret.rolling(window, min_periods=max(10, window // 3)).std()
    return roll_std * np.sqrt(TRADING_DAYS)


def compute_low_volatility(
    close:  pd.DataFrame,
    window: int = 63,
) -> pd.DataFrame:
    """
    Full low-volatility factor pipeline.

    Returns a (date × ticker) DataFrame of cross-sectionally z-scored
    low-vol scores.  Higher values = lower volatility = preferred long.
    """
    vol = trailing_realised_vol(close, window)

    # Winsorise to avoid extreme outliers (illiquid / recently listed stocks)
    vol_wins = vol.clip(
        lower=vol.quantile(0.01, axis=1),
        upper=vol.quantile(0.99, axis=1),
        axis=0,
    )

    # Negate: low vol → high score
    return cross_sectional_zscore(-vol_wins)
