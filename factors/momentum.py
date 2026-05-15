"""
Momentum factor: 12-minus-1 month cross-sectional return.

Signal definition
-----------------
For each stock on date t, compute the total return over [t-252, t-21]
(i.e. skip the most-recent month to avoid short-term reversal contamination).
The raw signal is then cross-sectionally rank-normalised to [-0.5, +0.5]
so it is comparable across time and suitable for equal-weight combination.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import MOM_LOOKBACK, MOM_SKIP


def raw_momentum(
    close: pd.DataFrame,
    lookback: int = MOM_LOOKBACK,
    skip: int = MOM_SKIP,
) -> pd.DataFrame:
    """
    Compute unadjusted 12-1 momentum returns for every date and stock.

    Parameters
    ----------
    close    : (date × ticker) adjusted-close price matrix
    lookback : total lookback in trading days  (~252 = 12 months)
    skip     : recent days to exclude         (~21  =  1 month)

    Returns
    -------
    DataFrame of the same shape as `close`, NaN for the first `lookback` rows.
    """
    if skip >= lookback:
        raise ValueError("`skip` must be smaller than `lookback`.")

    lagged_end   = close.shift(skip)           # t-1 month price
    lagged_start = close.shift(lookback)       # t-12 month price
    return lagged_end / lagged_start - 1


def cross_sectional_rank(signal: pd.DataFrame) -> pd.DataFrame:
    """
    Rank each stock within each date row; normalise to [-0.5, +0.5].

    Ties are averaged.  Rows with fewer than 2 non-NaN values return NaN.
    """
    ranks = signal.rank(axis=1, method="average", na_option="keep")
    count = ranks.notna().sum(axis=1)
    # normalise: 0 = bottom rank, 1 = top rank → shift to [-0.5, +0.5]
    normalised = ranks.sub(1).div(count.sub(1).replace(0, np.nan), axis=0).sub(0.5)
    return normalised


def compute_momentum(
    close: pd.DataFrame,
    lookback: int = MOM_LOOKBACK,
    skip: int = MOM_SKIP,
) -> pd.DataFrame:
    """
    Full momentum factor pipeline.

    Returns a (date × ticker) DataFrame of cross-sectionally rank-normalised
    12-1 momentum signals in [-0.5, +0.5].
    """
    raw  = raw_momentum(close, lookback, skip)
    return cross_sectional_rank(raw)
