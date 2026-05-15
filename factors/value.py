"""
Value factor: inverted price-to-book ratio.

Signal definition
-----------------
For each stock, we use the trailing price-to-book (P/B) ratio sourced from
yfinance .info.  Because yfinance only provides current-snapshot data, we
approximate the *historical* book-value per share as constant and scale it
by price changes to recover a time-series of P/B.

  Approximate P/B(t) = Price(t) / BookValue_current

This is a well-known simplification; a production system would replace it
with a proper point-in-time (PIT) fundamental feed.

The factor is then:
  Book-to-Price(t) = 1 / P/B(t)   (higher = cheaper = more value)

Cross-sectionally z-scored so it is comparable across time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def approximate_pb_series(
    close: pd.DataFrame,
    fundamentals: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a daily time-series of approximate P/B for each ticker.

    P/B(t) ≈ Price(t) / book_value_per_share_current
    where book_value_per_share_current = latest_price / latest_pb
    """
    # Latest available price (last row)
    latest_price = close.iloc[-1]

    fund = fundamentals.reindex(close.columns)
    pb_current = fund["priceToBook"]

    # book_value = latest_price / P/B_current  (constant approximation)
    book_value = latest_price / pb_current.replace(0, np.nan)

    # Historical P/B = Price(t) / book_value
    pb_series = close.div(book_value, axis=1)
    return pb_series


def cross_sectional_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise z-score: subtract cross-sectional mean, divide by std."""
    mean = df.mean(axis=1)
    std  = df.std(axis=1).replace(0, np.nan)
    return df.sub(mean, axis=0).div(std, axis=0)


def compute_value(
    close: pd.DataFrame,
    fundamentals: pd.DataFrame,
) -> pd.DataFrame:
    """
    Full value factor pipeline.

    Returns a (date × ticker) DataFrame of cross-sectionally z-scored
    book-to-price signals.  Higher values = more undervalued (better value).
    """
    pb = approximate_pb_series(close, fundamentals)

    # Winsorise extreme P/B values (cap at 1st / 99th percentile per date)
    pb_wins = pb.clip(
        lower=pb.quantile(0.01, axis=1), upper=pb.quantile(0.99, axis=1), axis=0
    )

    # Invert: B/P is the value factor (high B/P = cheap)
    bp = 1.0 / pb_wins.replace(0, np.nan)

    return cross_sectional_zscore(bp)
