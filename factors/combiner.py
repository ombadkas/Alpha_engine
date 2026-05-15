"""
Factor combination layer.

Normalises each factor to zero mean / unit variance cross-sectionally,
then combines them with configurable weights (default: equal weighting).

Output is a single composite alpha score per (date, ticker).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def zscore_signal(signal: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    """
    Cross-sectional z-score with outlier clipping.

    Parameters
    ----------
    signal : (date × ticker) factor values
    clip   : winsorise at ±clip standard deviations after z-scoring
    """
    mean = signal.mean(axis=1)
    std  = signal.std(axis=1).replace(0, np.nan)
    z = signal.sub(mean, axis=0).div(std, axis=0)
    return z.clip(-clip, clip)


def combine_factors(
    *factors: pd.DataFrame,
    weights: list[float] | None = None,
    align: bool = True,
) -> pd.DataFrame:
    """
    Weighted combination of factor DataFrames into a single composite score.

    Parameters
    ----------
    *factors : variable number of (date × ticker) factor DataFrames.
               Each is z-scored before combination.
    weights  : list of floats summing to 1.0.  Defaults to equal weights.
    align    : if True, forward-fill NaN factor values before combining
               (handles the initial warm-up period of individual factors).

    Returns
    -------
    (date × ticker) composite score DataFrame, cross-sectionally z-scored.
    """
    n = len(factors)
    if n == 0:
        raise ValueError("At least one factor required.")

    if weights is None:
        weights = [1.0 / n] * n
    else:
        weights = list(weights)
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0; got {sum(weights):.4f}.")
        if len(weights) != n:
            raise ValueError(f"Expected {n} weights, got {len(weights)}.")

    # Align shapes: use the intersection of all indices / columns
    idx   = factors[0].index
    cols  = factors[0].columns
    for f in factors[1:]:
        idx  = idx.intersection(f.index)
        cols = cols.intersection(f.columns)

    composite = pd.DataFrame(0.0, index=idx, columns=cols)

    for factor, w in zip(factors, weights):
        f = factor.reindex(index=idx, columns=cols)
        if align:
            f = f.ffill(limit=5)
        z = zscore_signal(f)
        composite = composite.add(w * z, fill_value=0.0)

    # Final cross-sectional z-score of the composite
    return zscore_signal(composite)


def build_composite(
    close: pd.DataFrame,
    fundamentals: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    3-factor composite (momentum + value + quality). Defaults to equal 1/3.
    """
    from factors.momentum import compute_momentum
    from factors.value    import compute_value
    from factors.quality  import compute_quality

    if weights is None:
        weights = {"momentum": 1 / 3, "value": 1 / 3, "quality": 1 / 3}

    mom  = compute_momentum(close)
    val  = compute_value(close, fundamentals)
    qual = compute_quality(close, fundamentals)

    return combine_factors(
        mom, val, qual,
        weights=[weights["momentum"], weights["value"], weights["quality"]],
    )


# ── Enhanced 4-factor composite ────────────────────────────────────────────

#: Momentum-heavy default weights that focus on pure price signals.
#: Value and quality are down-weighted because yfinance only provides
#: point-in-time (non-historical) fundamentals, making them noisy.
ENHANCED_WEIGHTS: dict[str, float] = {
    "momentum":    0.45,   # 12-1 month return — most robust price signal
    "low_vol":     0.30,   # low-volatility anomaly — pure price data
    "value":       0.13,   # B/P approximation — noisy, kept small
    "quality":     0.12,   # ROE proxy — noisy, kept small
}


def build_composite_enhanced(
    close:        pd.DataFrame,
    fundamentals: pd.DataFrame,
    weights:      dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    4-factor composite: momentum (45%) + low-vol (30%) + value (13%) + quality (12%).

    The low-volatility anomaly is computed entirely from price data, making it
    the most reliable additional signal given yfinance's snapshot-only fundamentals.

    Parameters
    ----------
    close        : adjusted-close price matrix
    fundamentals : fundamental data frame
    weights      : override default ENHANCED_WEIGHTS

    Returns
    -------
    composite score (date × ticker)
    """
    from factors.momentum       import compute_momentum
    from factors.value          import compute_value
    from factors.quality        import compute_quality
    from factors.low_volatility import compute_low_volatility

    w = weights if weights is not None else ENHANCED_WEIGHTS

    mom  = compute_momentum(close)
    lvol = compute_low_volatility(close)
    val  = compute_value(close, fundamentals)
    qual = compute_quality(close, fundamentals)

    return combine_factors(
        mom, lvol, val, qual,
        weights=[w["momentum"], w["low_vol"], w["value"], w["quality"]],
    )
