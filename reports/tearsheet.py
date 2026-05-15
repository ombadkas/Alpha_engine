"""
Research-grade tearsheet generator.

Produces a 6-panel matplotlib figure:
  1. Cumulative returns (strategy vs S&P 500 benchmark)
  2. Rolling 252-day Sharpe ratio
  3. Monthly return heatmap
  4. Drawdown profile
  5. Annual sector exposure (long vs short)
  6. Annual turnover bar chart

Call generate_tearsheet() to render and optionally save the figure.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless backend; swap to "TkAgg" for interactive

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from backtest.metrics import (
    drawdown_series,
    monthly_returns,
    rolling_sharpe,
    performance_summary,
    annual_turnover,
)
from config import PLOTS_DIR


# ── Colour palette ──────────────────────────────────────────────────────────
STRATEGY_COLOR  = "#1f77b4"
BENCHMARK_COLOR = "#aec7e8"
NEG_COLOR       = "#d62728"
POS_COLOR       = "#2ca02c"
DD_COLOR        = "#ff7f0e"
GRID_ALPHA      = 0.25


def _fmt_pct(ax, axis: str = "y") -> None:
    fmt = mtick.PercentFormatter(xmax=1, decimals=1)
    if axis == "y":
        ax.yaxis.set_major_formatter(fmt)
    else:
        ax.xaxis.set_major_formatter(fmt)


# ── Individual panels ───────────────────────────────────────────────────────

def _plot_cumulative(
    ax: plt.Axes,
    strategy: pd.Series,
    benchmark: pd.Series | None,
) -> None:
    cum_strat = (1 + strategy.fillna(0)).cumprod()
    ax.plot(cum_strat.index, cum_strat.values, color=STRATEGY_COLOR,
            lw=1.5, label="Strategy")
    if benchmark is not None:
        cum_bench = (1 + benchmark.fillna(0)).cumprod()
        # Align benchmark to strategy start
        cum_bench = cum_bench.reindex(cum_strat.index, method="ffill")
        ax.plot(cum_bench.index, cum_bench.values, color=BENCHMARK_COLOR,
                lw=1.2, linestyle="--", label="S&P 500")
    ax.set_title("Cumulative Return", fontweight="bold")
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.axhline(1.0, color="black", lw=0.5, linestyle=":")


def _plot_rolling_sharpe(ax: plt.Axes, strategy: pd.Series) -> None:
    rs = rolling_sharpe(strategy)
    ax.plot(rs.index, rs.values, color=STRATEGY_COLOR, lw=1.2)
    ax.axhline(0, color="black", lw=0.5, linestyle=":")
    ax.axhline(1, color=POS_COLOR, lw=0.5, linestyle="--", alpha=0.7)
    ax.fill_between(rs.index, rs.values, 0,
                    where=(rs > 0), alpha=0.15, color=POS_COLOR)
    ax.fill_between(rs.index, rs.values, 0,
                    where=(rs < 0), alpha=0.15, color=NEG_COLOR)
    ax.set_title("Rolling 252-day Sharpe", fontweight="bold")
    ax.set_ylabel("Sharpe Ratio")
    ax.grid(True, alpha=GRID_ALPHA)


def _plot_heatmap(ax: plt.Axes, strategy: pd.Series) -> None:
    try:
        pivot = monthly_returns(strategy)
    except Exception:
        ax.set_title("Monthly Returns (insufficient data)")
        return

    # Fill missing months with NaN colour
    cmap = plt.get_cmap("RdYlGn")
    vmax = max(abs(pivot.values[~np.isnan(pivot.values)]).max(), 0.01) if pivot.notna().any().any() else 0.05

    im = ax.imshow(pivot.values, cmap=cmap, aspect="auto",
                   vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, format=lambda x, _: f"{x:.1%}", shrink=0.8)

    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_title("Monthly Returns Heatmap", fontweight="bold")

    # Annotate cells
    for (r, c), val in np.ndenumerate(pivot.values):
        if not np.isnan(val):
            ax.text(c, r, f"{val:.1%}", ha="center", va="center",
                    fontsize=5, color="black")


def _plot_drawdown(ax: plt.Axes, strategy: pd.Series) -> None:
    dd = drawdown_series(strategy)
    ax.fill_between(dd.index, dd.values, 0, alpha=0.7, color=DD_COLOR)
    ax.set_title("Drawdown Profile", fontweight="bold")
    ax.set_ylabel("Drawdown")
    _fmt_pct(ax)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.axhline(0, color="black", lw=0.5)


def _plot_sector_exposure(
    ax: plt.Axes,
    positions: pd.DataFrame | None,
) -> None:
    """Plot annual long/short gross exposure by sector (if sector data available)."""
    ax.set_title("Portfolio Exposure (long / short)", fontweight="bold")

    if positions is None or positions.empty:
        ax.text(0.5, 0.5, "No position data available",
                ha="center", va="center", transform=ax.transAxes)
        return

    # Without sector mapping, show long vs short gross exposure over time
    long_exp  = positions.clip(lower=0).sum(axis=1).resample("YE").mean()
    short_exp = positions.clip(upper=0).sum(axis=1).abs().resample("YE").mean()

    years = [str(y.year) for y in long_exp.index]
    x = np.arange(len(years))
    width = 0.35

    ax.bar(x - width / 2, long_exp.values,  width, label="Long",  color=POS_COLOR, alpha=0.8)
    ax.bar(x + width / 2, short_exp.values, width, label="Short", color=NEG_COLOR, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Gross Exposure")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=GRID_ALPHA, axis="y")


def _plot_turnover(ax: plt.Axes, turnover: pd.Series | None) -> None:
    ax.set_title("Annual Turnover", fontweight="bold")

    if turnover is None or turnover.empty:
        ax.text(0.5, 0.5, "No turnover data available",
                ha="center", va="center", transform=ax.transAxes)
        return

    ann_to = turnover.resample("YE").sum()  # daily → annual
    years  = [str(y.year) for y in ann_to.index]
    x = np.arange(len(years))

    ax.bar(x, ann_to.values, color=STRATEGY_COLOR, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Annual One-Way Turnover")
    ax.grid(True, alpha=GRID_ALPHA, axis="y")
    _fmt_pct(ax)


# ── Stats banner ────────────────────────────────────────────────────────────

def _add_stats_banner(
    fig: plt.Figure,
    strategy: pd.Series,
    benchmark: pd.Series | None,
    turnover: pd.Series | None,
) -> None:
    stats = performance_summary(strategy, benchmark, turnover)

    lines = [
        f"Ann. Return: {stats.get('Annualised Return', float('nan')):.2%}",
        f"Ann. Vol: {stats.get('Annualised Vol', float('nan')):.2%}",
        f"Sharpe: {stats.get('Sharpe Ratio', float('nan')):.2f}",
        f"Max DD: {stats.get('Max Drawdown', float('nan')):.2%}",
        f"Calmar: {stats.get('Calmar Ratio', float('nan')):.2f}",
    ]
    if "Information Ratio" in stats:
        lines.append(f"IR: {stats['Information Ratio']:.2f}")
    if "Annual Turnover" in stats:
        lines.append(f"Turnover: {stats['Annual Turnover']:.2%}")

    banner = "   |   ".join(lines)
    fig.text(0.5, 0.97, banner, ha="center", va="top", fontsize=9,
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="grey", alpha=0.8))


# ── Main tearsheet ──────────────────────────────────────────────────────────

def generate_tearsheet(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    positions: pd.DataFrame | None = None,
    turnover: pd.Series | None = None,
    title: str = "Multi-Factor Alpha Engine — Performance Tearsheet",
    save_path: Path | str | None = None,
    show: bool = False,
) -> plt.Figure:
    """
    Generate a research-grade 6-panel tearsheet.

    Parameters
    ----------
    strategy_returns  : daily net returns of the L/S strategy
    benchmark_returns : daily returns of the benchmark (e.g. SPY)
    positions         : (date × ticker) holdings DataFrame
    turnover          : daily one-way portfolio turnover
    title             : figure super-title
    save_path         : if provided, save figure to this path
    show              : if True, call plt.show()

    Returns
    -------
    matplotlib Figure object
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        fig = plt.figure(figsize=(18, 20))
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.99)

        gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.30,
                              top=0.94, bottom=0.05)

        ax1 = fig.add_subplot(gs[0, :])          # full width – cumulative
        ax2 = fig.add_subplot(gs[1, 0])          # rolling Sharpe
        ax3 = fig.add_subplot(gs[1, 1])          # monthly heatmap
        ax4 = fig.add_subplot(gs[2, 0])          # drawdown
        ax5 = fig.add_subplot(gs[2, 1])          # exposure / turnover

        _plot_cumulative(ax1, strategy_returns, benchmark_returns)
        _plot_rolling_sharpe(ax2, strategy_returns)
        _plot_heatmap(ax3, strategy_returns)
        _plot_drawdown(ax4, strategy_returns)

        if turnover is not None:
            _plot_turnover(ax5, turnover)
        else:
            _plot_sector_exposure(ax5, positions)

        _add_stats_banner(fig, strategy_returns, benchmark_returns, turnover)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Tearsheet saved -> {save_path}")

    if show:
        plt.show()

    return fig
