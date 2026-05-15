"""
Performance metrics for the alpha engine.

All functions accept a pd.Series of daily returns (fractional, not percent).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS = 252


# ── Core metrics ───────────────────────────────────────────────────────────

def annualised_return(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Compound annualised growth rate."""
    total = (1 + returns.dropna()).prod()
    n = len(returns.dropna())
    if n == 0:
        return np.nan
    return float(total ** (periods / n) - 1)


def annualised_volatility(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    return float(returns.dropna().std() * np.sqrt(periods))


def sharpe_ratio(
    returns: pd.Series,
    rf: float = 0.0,
    periods: int = TRADING_DAYS,
) -> float:
    """Annualised Sharpe ratio. rf is the annualised risk-free rate."""
    daily_rf = (1 + rf) ** (1 / periods) - 1
    excess   = returns.dropna() - daily_rf
    vol      = excess.std()
    if vol == 0:
        return np.nan
    return float(excess.mean() / vol * np.sqrt(periods))


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (negative number)."""
    wealth = (1 + returns.dropna()).cumprod()
    peak   = wealth.cummax()
    dd     = (wealth - peak) / peak
    return float(dd.min())


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Running drawdown series."""
    wealth = (1 + returns.fillna(0)).cumprod()
    peak   = wealth.cummax()
    return (wealth - peak) / peak


def calmar_ratio(returns: pd.Series) -> float:
    """Annualised return divided by abs(max drawdown)."""
    mdd = max_drawdown(returns)
    if mdd == 0:
        return np.nan
    return annualised_return(returns) / abs(mdd)


def information_ratio(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods: int = TRADING_DAYS,
) -> float:
    """Annualised information ratio against a benchmark."""
    active = strategy_returns.sub(benchmark_returns).dropna()
    tracking_error = active.std() * np.sqrt(periods)
    if tracking_error == 0:
        return np.nan
    return float(active.mean() * periods / tracking_error)


def rolling_sharpe(
    returns: pd.Series,
    window: int = TRADING_DAYS,
    rf: float = 0.0,
) -> pd.Series:
    """Rolling annualised Sharpe ratio."""
    daily_rf = (1 + rf) ** (1 / TRADING_DAYS) - 1
    excess   = returns - daily_rf
    roll_mean = excess.rolling(window).mean()
    roll_std  = excess.rolling(window).std()
    return (roll_mean / roll_std.replace(0, np.nan)) * np.sqrt(TRADING_DAYS)


def annual_turnover(turnover: pd.Series) -> float:
    """Average annual one-way portfolio turnover."""
    return float(turnover.mean() * TRADING_DAYS)


def monthly_returns(returns: pd.Series) -> pd.DataFrame:
    """
    Reshape daily returns into a (year × month) pivot table of monthly returns.
    """
    monthly = (1 + returns).resample("ME").prod() - 1
    pivot = monthly.groupby([monthly.index.year, monthly.index.month]).first().unstack()
    pivot.index.name  = "Year"
    pivot.columns.name = "Month"
    pivot.columns = [
        "Jan","Feb","Mar","Apr","May","Jun",
        "Jul","Aug","Sep","Oct","Nov","Dec",
    ][: pivot.shape[1]]
    return pivot


# ── CAPM decomposition ────────────────────────────────────────────────────

def capm_alpha_beta(
    strategy_returns:  pd.Series,
    benchmark_returns: pd.Series,
) -> tuple[float, float]:
    """
    OLS regression of strategy returns on benchmark returns.

    Returns
    -------
    (annualised_alpha, beta)

    CAPM alpha is the regression intercept × 252.  This is the correct
    benchmark-adjusted excess return for any strategy with non-zero beta,
    including long-only portfolios.
    """
    s, b = strategy_returns.align(benchmark_returns, join="inner")
    mask = s.notna() & b.notna()
    s, b = s[mask].values, b[mask].values

    if len(s) < 30:
        return np.nan, np.nan

    # OLS via normal equations: [alpha_daily, beta] = pinv(X) @ y
    X    = np.column_stack([np.ones(len(b)), b])
    coef = np.linalg.lstsq(X, s, rcond=None)[0]
    daily_alpha, beta = float(coef[0]), float(coef[1])
    return daily_alpha * TRADING_DAYS, beta


# ── Drawdown-aware return series ───────────────────────────────────────────

def apply_drawdown_control(
    returns:            pd.Series,
    trigger:            float = -0.10,   # start scaling at -10% drawdown
    floor:              float = -0.20,   # fully halved at -20%
    min_scale:          float = 0.30,    # minimum scale factor
) -> pd.Series:
    """
    Linearly reduce exposure as the portfolio enters drawdown.

    At drawdown = trigger (e.g. -10%), scale = 1.0.
    At drawdown = floor   (e.g. -20%), scale = min_scale.
    All scale factors are lagged by one day (no look-ahead).

    Parameters
    ----------
    trigger   : drawdown level at which scaling begins
    floor     : drawdown level at which minimum scale is reached
    min_scale : minimum portfolio scale (never goes fully flat)
    """
    dd = drawdown_series(returns)

    # Linear interpolation between trigger and floor
    scale = 1.0 + (min_scale - 1.0) * ((dd - trigger) / (floor - trigger)).clip(0, 1)
    scale = scale.shift(1).fillna(1.0)   # strict no look-ahead
    return returns * scale


# ── Summary table ──────────────────────────────────────────────────────────

def performance_summary(
    strategy_returns:  pd.Series,
    benchmark_returns: pd.Series | None = None,
    turnover:          pd.Series | None = None,
) -> pd.Series:
    """Return a labelled Series of key performance statistics."""
    stats: dict[str, float] = {
        "Annualised Return": annualised_return(strategy_returns),
        "Annualised Vol":    annualised_volatility(strategy_returns),
        "Sharpe Ratio":      sharpe_ratio(strategy_returns),
        "Max Drawdown":      max_drawdown(strategy_returns),
        "Calmar Ratio":      calmar_ratio(strategy_returns),
    }
    if benchmark_returns is not None:
        alpha_capm, beta = capm_alpha_beta(strategy_returns, benchmark_returns)
        stats["Beta"]             = beta
        stats["CAPM Alpha (ann)"] = alpha_capm          # regression-based, correct measure
        stats["Information Ratio"] = information_ratio(strategy_returns, benchmark_returns)
        # Raw return difference kept for transparency
        bench_ret = annualised_return(benchmark_returns)
        stats["Excess Return vs Bench"] = stats["Annualised Return"] - bench_ret

    if turnover is not None:
        stats["Annual Turnover"] = annual_turnover(turnover)

    return pd.Series(stats).round(4)
