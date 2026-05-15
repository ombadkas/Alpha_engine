"""
Risk management overlays for the alpha engine.

Seven strategies compared:

  baseline        — L/S equal-weight, monthly, 3-factor composite.
  signal_buffer   — L/S quarterly rebalancing; cuts turnover ~60%.
  vol_weighted    — L/S inverse-vol weighting within quintile.
  vol_target      — L/S + daily vol-targeting (10% target).
  combined        — L/S quarterly + inv-vol + vol-target.

  ── Enhanced (4-factor composite: momentum 45% + low-vol 30% + value 13% + quality 12%) ──
  long_only_vt    — Long top quintile only + vol-target at benchmark vol (15%).
                    Has ~1.0 market beta, designed to beat SPY on CAPM alpha.
  drawdown_aware  — L/S vol-target + linear drawdown de-risking; minimises
                    max drawdown by scaling exposure as portfolio falls.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from config import N_QUINTILES, REBAL_FREQ

TRADING_DAYS   = 252
QUARTERLY_FREQ = "BQE"


# ── Configuration ──────────────────────────────────────────────────────────

@dataclass
class RiskConfig:
    strategy: Literal[
        "baseline", "signal_buffer", "vol_weighted", "vol_target", "combined",
        "long_only_vt", "drawdown_aware",
    ] = "combined"

    # Volatility weighting
    vol_window: int   = 63
    vol_floor:  float = 0.05

    # Volatility targeting
    target_vol:        float = 0.10
    vol_target_window: int   = 63
    max_leverage:      float = 1.5
    min_leverage:      float = 0.20

    # Drawdown control (drawdown_aware only)
    dd_trigger:   float = -0.10   # start scaling at -10% drawdown
    dd_floor:     float = -0.20   # fully scaled at -20%
    dd_min_scale: float = 0.30    # minimum scale

    n_quintiles: int = N_QUINTILES

    @property
    def rebal_freq(self) -> str:
        if self.strategy in ("signal_buffer", "combined"):
            return QUARTERLY_FREQ
        return REBAL_FREQ

    @property
    def use_vol_weight(self) -> bool:
        return self.strategy in ("vol_weighted", "combined")

    @property
    def use_vol_target(self) -> bool:
        return self.strategy in ("vol_target", "combined", "long_only_vt", "drawdown_aware")

    @property
    def long_only(self) -> bool:
        return self.strategy == "long_only_vt"

    @property
    def use_dd_control(self) -> bool:
        return self.strategy == "drawdown_aware"


# ── Weight builders ────────────────────────────────────────────────────────

def _quintile_cut(scores: pd.Series, n: int) -> pd.Series:
    valid = scores.dropna()
    if valid.empty:
        return pd.Series(dtype=float)
    return pd.qcut(valid, n, labels=False, duplicates="drop")


def _score_to_weights_eq(scores: pd.Series, n: int = N_QUINTILES) -> pd.Series:
    """Equal-weight L/S: long top quintile, short bottom quintile."""
    q = _quintile_cut(scores, n)
    w = pd.Series(0.0, index=scores.index)
    top, bot = q[q == n - 1].index, q[q == 0].index
    if len(top): w[top]    = +1.0 / len(top)
    if len(bot): w[bot]    = -1.0 / len(bot)
    return w


def _score_to_weights_long_only(scores: pd.Series, n: int = N_QUINTILES) -> pd.Series:
    """Equal-weight long-only: top quintile, fully invested (no short)."""
    q = _quintile_cut(scores, n)
    w = pd.Series(0.0, index=scores.index)
    top = q[q == n - 1].index
    if len(top): w[top] = 1.0 / len(top)
    return w


def _score_to_weights_vol(
    scores: pd.Series, avol: pd.Series,
    vol_floor: float = 0.05, n: int = N_QUINTILES,
) -> pd.Series:
    """Inverse-vol weighted L/S."""
    q = _quintile_cut(scores, n)
    w = pd.Series(0.0, index=scores.index)
    for members, sign in [(q[q == n - 1].index, +1), (q[q == 0].index, -1)]:
        if not len(members): continue
        inv = 1.0 / avol.reindex(members).clip(lower=vol_floor).fillna(vol_floor)
        w[members] = sign * inv / inv.sum()
    return w


def _trailing_vol(close: pd.DataFrame, dt: pd.Timestamp, window: int) -> pd.Series:
    loc   = close.index.get_loc(dt)
    start = max(0, loc - window)
    return close.iloc[start:loc].pct_change(fill_method=None).std() * np.sqrt(TRADING_DAYS)


def build_risk_weight_matrix(
    composite: pd.DataFrame,
    close:     pd.DataFrame,
    cfg:       RiskConfig,
) -> pd.DataFrame:
    """Build target-weight matrix; only rebalancing dates have non-NaN rows."""
    rebal_dates = (
        composite.resample(cfg.rebal_freq).last().index
        .intersection(composite.index)
    )
    wm = pd.DataFrame(np.nan, index=composite.index, columns=composite.columns)

    for dt in rebal_dates:
        scores = composite.loc[dt]
        if cfg.long_only:
            row = _score_to_weights_long_only(scores, cfg.n_quintiles)
        elif cfg.use_vol_weight:
            avol = _trailing_vol(close, dt, cfg.vol_window).reindex(composite.columns)
            row  = _score_to_weights_vol(scores, avol, cfg.vol_floor, cfg.n_quintiles)
        else:
            row  = _score_to_weights_eq(scores, cfg.n_quintiles)
        wm.loc[dt] = row

    return wm


# ── Post-simulation overlays ───────────────────────────────────────────────

def apply_vol_target(returns: pd.Series, cfg: RiskConfig) -> pd.Series:
    """Scale daily returns to target cfg.target_vol. Strictly lagged."""
    if not cfg.use_vol_target:
        return returns
    realised = returns.rolling(cfg.vol_target_window).std() * np.sqrt(TRADING_DAYS)
    scale    = (cfg.target_vol / realised.shift(1)).clip(cfg.min_leverage, cfg.max_leverage)
    return returns * scale.fillna(cfg.min_leverage)


def apply_drawdown_control(returns: pd.Series, cfg: RiskConfig) -> pd.Series:
    """
    Linearly reduce exposure as portfolio enters drawdown.

    scale = 1.0 at dd = cfg.dd_trigger (e.g. -10%)
    scale = cfg.dd_min_scale at dd = cfg.dd_floor (e.g. -20%)
    All scaling is lagged one day (no look-ahead).
    """
    if not cfg.use_dd_control:
        return returns
    from backtest.metrics import drawdown_series
    dd    = drawdown_series(returns)
    t, f  = cfg.dd_trigger, cfg.dd_floor
    scale = 1.0 + (cfg.dd_min_scale - 1.0) * ((dd - t) / (f - t)).clip(0, 1)
    return returns * scale.shift(1).fillna(1.0)


# ── End-to-end strategy runner ─────────────────────────────────────────────

def run_risk_strategy(
    close:        pd.DataFrame,
    composite:    pd.DataFrame,
    cfg:          RiskConfig,
    fees:         float = 0.0005,
    # Long-only strategies need benchmark vol target, not 10%
    bench_annual_vol: float | None = None,
) -> dict:
    """Full simulation for one RiskConfig."""
    from backtest.engine import simulate_portfolio
    from config import BACKTEST_START

    close     = close.loc[BACKTEST_START:]
    composite = composite.loc[BACKTEST_START:]
    common    = close.columns.intersection(composite.columns)
    close, composite = close[common], composite[common]

    # Long-only targets benchmark vol so we don't under-invest
    if cfg.long_only and bench_annual_vol is not None:
        cfg = RiskConfig(**{**cfg.__dict__,
                            "target_vol":    bench_annual_vol,
                            "max_leverage":  1.0,
                            "min_leverage":  0.5})

    wm     = build_risk_weight_matrix(composite, close, cfg)
    result = simulate_portfolio(close, wm, fees=fees)

    returns = result["returns"]
    returns = apply_vol_target(returns, cfg)
    returns = apply_drawdown_control(returns, cfg)
    result["returns"] = returns
    result["cfg"]     = cfg
    result["label"]   = cfg.strategy.replace("_", " ").title()
    return result


# ── Comparison runner ──────────────────────────────────────────────────────

RISK_STRATEGIES: list[RiskConfig] = [
    RiskConfig(strategy="baseline"),
    RiskConfig(strategy="vol_target"),
    RiskConfig(strategy="long_only_vt"),
    RiskConfig(strategy="drawdown_aware"),
    RiskConfig(strategy="combined"),
]


def compare_strategies(
    close:             pd.DataFrame,
    composite:         pd.DataFrame,
    benchmark_returns: pd.Series | None = None,
    fees:              float = 0.0005,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Run all strategies and return a comparison summary."""
    from backtest.metrics import performance_summary, annualised_volatility

    bench_vol = (
        annualised_volatility(benchmark_returns)
        if benchmark_returns is not None else None
    )

    all_results: dict[str, dict] = {}
    rows: list[dict] = []

    for cfg in RISK_STRATEGIES:
        print(f"  Running: {cfg.strategy} ...", flush=True)
        r = run_risk_strategy(close, composite, cfg, fees=fees,
                              bench_annual_vol=bench_vol)
        all_results[cfg.strategy] = r
        stats = performance_summary(
            r["returns"].dropna(), benchmark_returns, r["turnover"]
        ).to_dict()
        stats["Strategy"] = cfg.strategy.replace("_", " ").title()
        rows.append(stats)

    summary = (
        pd.DataFrame(rows)
        .set_index("Strategy")
        .sort_values("Sharpe Ratio", ascending=False)
    )
    return summary, all_results


def best_strategy(summary_df: pd.DataFrame) -> str:
    return summary_df["Sharpe Ratio"].idxmax()
