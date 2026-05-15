"""
Multi-Factor Equity Alpha Engine — main entry-point.

Usage
-----
  python main.py                    # full pipeline (download + backtest + tearsheet)
  python main.py --no-download      # use cached data
  python main.py --no-risk          # skip risk comparison, use baseline only
  python main.py --wf               # include walk-forward validation
  python main.py --show             # open tearsheet in a window
  python main.py --weights 0.5 0.3 0.2  # custom factor weights
"""
from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("alpha_engine")

_PCT  = {"Drawdown", "Return", "Vol", "Turnover", "Alpha", "Excess"}
_FMT  = lambda label, val: (
    f"  {label:<30} {val:>9.2%}" if any(k in label for k in _PCT)
    else f"  {label:<30} {val:>9.3f}"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-Factor Equity Alpha Engine")
    p.add_argument("--no-download", action="store_true",
                   help="Skip data download; use cached parquet files")
    p.add_argument("--no-risk", action="store_true",
                   help="Skip risk-strategy comparison; run baseline only")
    p.add_argument("--wf", action="store_true",
                   help="Run walk-forward OOS validation on best strategy")
    p.add_argument("--show", action="store_true",
                   help="Display tearsheet interactively")
    p.add_argument("--no-enhanced", action="store_true",
                   help="Use 3-factor composite instead of 4-factor enhanced")
    return p.parse_args()


def _print_summary(title: str, summary: pd.Series) -> None:
    print(f"\n{'=' * 55}")
    print(title)
    print("=" * 55)
    for label, val in summary.items():
        print(_FMT(label, val))
    print("=" * 55)


def main() -> None:
    args = parse_args()

    # ── 1. Data ingestion ────────────────────────────────────────────────────
    from data.ingest import run_ingestion
    log.info("=== Step 1: Data Ingestion ===")
    prices, fundamentals = run_ingestion(force=not args.no_download)
    log.info("Prices: %s  |  Fundamentals: %s", prices.shape, fundamentals.shape)

    # ── 2. Factor computation ────────────────────────────────────────────────
    log.info("=== Step 2: Factor Computation ===")
    if args.no_enhanced:
        from factors.combiner import build_composite
        composite = build_composite(prices, fundamentals)
        log.info("Using 3-factor composite (momentum / value / quality)")
    else:
        from factors.combiner import build_composite_enhanced
        composite = build_composite_enhanced(prices, fundamentals)
        log.info("Using 4-factor composite (momentum 45%% / low-vol 30%% / value 13%% / quality 12%%)")
    log.info("Composite score: %s  (non-NaN: %d%%)",
             composite.shape, int(composite.notna().mean().mean() * 100))

    # ── 3. Benchmark returns ─────────────────────────────────────────────────
    from config import BACKTEST_START, END_DATE, BENCH_TICKER, FEES
    log.info("=== Step 3: Benchmark Data ===")
    try:
        spy = yf.download(BENCH_TICKER, start=BACKTEST_START, end=END_DATE,
                          auto_adjust=True, progress=False)
        bench_returns = spy["Close"].squeeze().pct_change().dropna()
        bench_returns.name = BENCH_TICKER
    except Exception as exc:
        log.warning("Could not download benchmark: %s — continuing without.", exc)
        bench_returns = None

    # ── 4. Risk strategy comparison ──────────────────────────────────────────
    from backtest.metrics import performance_summary

    if args.no_risk:
        # Fast path: baseline only
        from backtest.engine import run_backtest
        log.info("=== Step 4: Baseline Backtest ===")
        result = run_backtest(prices, composite)
        best_result = result
        best_result["label"] = "Baseline"
        comparison_df = None
    else:
        from backtest.risk import compare_strategies, best_strategy
        log.info("=== Step 4: Risk Strategy Comparison ===")
        print()
        comparison_df, all_results = compare_strategies(
            prices, composite, bench_returns, fees=FEES
        )

        # Print comparison table
        print("\n" + "=" * 75)
        print("RISK STRATEGY COMPARISON  (sorted by Sharpe)")
        print("=" * 75)
        display_cols = [
            "Annualised Return", "Annualised Vol", "Sharpe Ratio",
            "Max Drawdown", "Calmar Ratio", "Annual Turnover",
        ]
        if bench_returns is not None:
            display_cols += ["Beta", "CAPM Alpha (ann)", "Excess Return vs Bench"]
        tbl = comparison_df[[c for c in display_cols if c in comparison_df.columns]].copy()
        fmt_map = {
            "Annualised Return":      "{:.2%}",
            "Annualised Vol":         "{:.2%}",
            "Sharpe Ratio":           "{:.3f}",
            "Max Drawdown":           "{:.2%}",
            "Calmar Ratio":           "{:.3f}",
            "Annual Turnover":        "{:.0%}",
            "Beta":                   "{:.3f}",
            "CAPM Alpha (ann)":       "{:.2%}",
            "Excess Return vs Bench": "{:.2%}",
        }
        for col, fmt in fmt_map.items():
            if col in tbl.columns:
                tbl[col] = tbl[col].apply(lambda v: fmt.format(v) if pd.notna(v) else "---")
        print(tbl.to_string())
        print("=" * 90)

        winner = best_strategy(comparison_df)
        print(f"\n>>  Best strategy by Sharpe: {winner}\n")
        best_result = all_results[winner.lower().replace(" ", "_")]

    # ── 5. Best-strategy summary ─────────────────────────────────────────────
    log.info("=== Step 5: Performance Summary (%s) ===", best_result.get("label", ""))
    best_returns  = best_result["returns"].dropna()
    best_turnover = best_result["turnover"]
    best_positions = best_result.get("positions")

    summary = performance_summary(best_returns, bench_returns, best_turnover)
    _print_summary(
        f"BEST STRATEGY: {best_result.get('label', '').upper()}",
        summary,
    )

    # ── 6. Walk-forward validation (optional) ────────────────────────────────
    oos_returns = None
    if args.wf:
        from backtest.walk_forward import run_walk_forward, fold_summary_table
        log.info("=== Step 6: Walk-Forward Validation ===")
        oos_returns, folds = run_walk_forward(prices, composite, bench_returns)
        fold_tbl = fold_summary_table(folds)
        log.info("\nWalk-Forward Fold Summary:\n%s", fold_tbl.to_string())

    # ── 7. Tearsheet ─────────────────────────────────────────────────────────
    from reports.tearsheet import generate_tearsheet
    from config import PLOTS_DIR
    log.info("=== Step 7: Tearsheet ===")

    display_returns = oos_returns if oos_returns is not None else best_returns

    # Generate tearsheet for best strategy
    save_path = PLOTS_DIR / f"tearsheet_{best_result.get('label','strategy').lower().replace(' ','_')}.png"
    generate_tearsheet(
        strategy_returns=display_returns,
        benchmark_returns=bench_returns,
        positions=best_positions,
        turnover=best_turnover,
        title=f"Multi-Factor Alpha Engine — {best_result.get('label','')} Tearsheet",
        save_path=save_path,
        show=args.show,
    )

    # Also generate a comparison chart if multiple strategies were run
    if comparison_df is not None:
        _save_comparison_chart(all_results, bench_returns, PLOTS_DIR / "strategy_comparison.png")

    log.info("Done.  Tearsheet -> %s", save_path)


def _save_comparison_chart(
    all_results: dict,
    bench_returns: pd.Series | None,
    save_path,
) -> None:
    """Overlay cumulative returns for all strategies on one chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                   gridspec_kw={"height_ratios": [3, 1]})

    colors = ["#555555", "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    styles = ["--", "-", "-", "-", "-"]

    for (name, result), color, ls in zip(all_results.items(), colors, styles):
        rets = result["returns"].dropna()
        cum  = (1 + rets).cumprod()
        lw   = 2.5 if name == "combined" else 1.2
        ax1.plot(cum.index, cum.values, label=result["label"], color=color,
                 lw=lw, linestyle=ls)

    if bench_returns is not None:
        cum_b = (1 + bench_returns.fillna(0)).cumprod()
        ax1.plot(cum_b.index, cum_b.values, label="S&P 500", color="lightblue",
                 lw=1.0, linestyle=":", alpha=0.8)

    ax1.axhline(1.0, color="black", lw=0.5, linestyle=":")
    ax1.set_title("Risk Strategy Comparison — Cumulative Returns", fontweight="bold")
    ax1.set_ylabel("Growth of $1")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.25)

    # Turnover comparison
    from backtest.metrics import annual_turnover
    labels = [r["label"] for r in all_results.values()]
    to_vals = [annual_turnover(r["turnover"]) for r in all_results.values()]
    bars = ax2.bar(labels, to_vals, color=colors, alpha=0.8)
    ax2.set_title("Annual Turnover by Strategy", fontweight="bold")
    ax2.set_ylabel("One-Way Turnover")
    import matplotlib.ticker as mtick
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax2.grid(True, alpha=0.25, axis="y")
    for bar, val in zip(bars, to_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                 f"{val:.0%}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    import pathlib
    pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Comparison chart saved -> {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
