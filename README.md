# Multi-Factor Equity Alpha Engine

Systematic equity strategy built on a 4-factor composite signal across the S&P 500.
Compares five risk overlays and selects the best by Sharpe. Produces a full tearsheet.

---

## Structure

```
alpha-engine/
├── data/
│   ├── ingest.py
│   ├── raw/
│   └── processed/
├── factors/
│   ├── momentum.py
│   ├── low_volatility.py
│   ├── value.py
│   ├── quality.py
│   └── combiner.py
├── backtest/
│   ├── engine.py
│   ├── metrics.py
│   ├── risk.py
│   └── walk_forward.py
├── reports/
│   ├── tearsheet.py
│   └── plots/
├── tests/
├── .github/workflows/weekly_run.yml
├── main.py
└── config.py
```

---

## How it works

**Data** — S&P 500 tickers from Wikipedia, OHLCV + fundamentals via yfinance, stored as parquet.

**Factors** — Four signals, computed cross-sectionally each month:

| Factor | Weight | Signal |
|--------|--------|--------|
| Momentum | 45% | 12-1 month return, rank-normalised |
| Low Volatility | 30% | Trailing 63-day realised vol, negated |
| Value | 13% | Book-to-price, winsorised |
| Quality | 12% | ROE + accruals proxy |

Momentum and low-vol are up-weighted because both derive purely from price data. Value and quality use yfinance snapshot fundamentals (non-historical), so they contribute less.

**Portfolio** — Long top quintile. Monthly rebalancing. 5 bps one-way transaction costs.

**Risk overlays** — Five strategies compared on each run:

| Strategy | What it does |
|----------|-------------|
| `baseline` | Equal-weight L/S, monthly |
| `vol_target` | L/S + daily vol-targeting to 10% |
| `long_only_vt` | Long top quintile + vol-target at benchmark vol |
| `drawdown_aware` | L/S + vol-target + linear de-risking below −10% DD |
| `combined` | Quarterly L/S + inv-vol weights + vol-target |

**Walk-forward** — 3-year train, 1-year test, rolling forward. Strictly out-of-sample.

---

## Performance (2015–2024)

Best strategy: **Long Only VT**

<!-- STATS_START -->
_Last updated: 2026-05-15_

| Metric | Value |
|--------|-------|
| Annualised Return | 12.62% |
| Annualised Vol | 14.69% |
| Sharpe Ratio | 0.883 |
| Max Drawdown | -24.71% |
| Calmar Ratio | 0.511 |
| Beta | 0.715 |
| CAPM Alpha | +3.07% |
| Information Ratio | -0.097 |
| Excess Return vs Bench | -0.46% |
| Annual Turnover | 330% |
<!-- STATS_END -->

> Updated automatically every Sunday via GitHub Actions.

### All strategies

| Strategy | Sharpe | Return | Max DD | Beta | CAPM Alpha | Turnover |
|----------|--------|--------|--------|------|------------|----------|
| Long Only VT | 0.883 | 12.62% | -24.7% | 0.72 | +3.07% | 330% |
| Vol Target | 0.029 | -0.26% | -27.2% | -0.14 | +2.27% | 592% |
| Combined | -0.092 | -1.52% | -35.3% | -0.12 | +0.65% | 368% |
| Drawdown Aware | -0.176 | -1.57% | -29.1% | -0.11 | +0.16% | 592% |
| Baseline | -0.264 | -6.70% | -62.1% | -0.33 | -0.49% | 592% |

---

## Usage

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# Full run (downloads data + backtests + tearsheet)
python main.py

# Cached data
python main.py --no-download

# With walk-forward validation
python main.py --no-download --wf

# 3-factor composite instead of 4
python main.py --no-enhanced

pytest tests/ -v
```

---

## Key config (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `START_DATE` | `2014-01-01` | Download start (extra year for momentum lookback) |
| `BACKTEST_START` | `2015-01-01` | Backtest start |
| `END_DATE` | `2024-12-31` | Backtest end |
| `FEES_BPS` | `5` | One-way transaction cost (bps) |
| `N_QUINTILES` | `5` | Portfolio construction quintiles |
| `REBAL_FREQ` | `BME` | Monthly rebalancing |
| `TRAIN_YEARS` | `3` | Walk-forward training window |
| `TEST_YEARS` | `1` | Walk-forward test window |

---

## Limitations

- **Survivorship bias** — Wikipedia tickers are current constituents only; delisted stocks excluded.
- **Point-in-time fundamentals** — yfinance returns the latest P/B and ROE, not historical snapshots. Value and quality signals have mild look-ahead bias.
- **Transaction costs** — fixed at 5 bps; real costs vary with liquidity and order size.

---

## License

MIT
