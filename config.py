"""Central configuration for the Multi-Factor Equity Alpha Engine."""
from pathlib import Path

ROOT = Path(__file__).parent

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_RAW       = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
PLOTS_DIR      = ROOT / "reports" / "plots"

# ── Universe & dates ───────────────────────────────────────────────────────
START_DATE       = "2014-01-01"   # extra year for 12-month lookback
END_DATE         = "2024-12-31"
BACKTEST_START   = "2015-01-01"
MAX_TICKERS      = 500            # cap for speed; set None for full S&P 500

# ── Factor parameters ──────────────────────────────────────────────────────
MOM_LOOKBACK  = 252   # trading days (~12 months)
MOM_SKIP      = 21    # skip most-recent month (21 trading days)

# ── Portfolio construction ─────────────────────────────────────────────────
N_QUINTILES      = 5
FEES_BPS         = 5           # one-way transaction cost in basis points
FEES             = FEES_BPS / 10_000
REBAL_FREQ       = "BME"       # business month-end rebalancing

# ── Walk-forward ───────────────────────────────────────────────────────────
TRAIN_YEARS = 3
TEST_YEARS  = 1

# ── Reporting ──────────────────────────────────────────────────────────────
RISK_FREE_RATE = 0.0      # annualised; set to 0 for simplicity
BENCH_TICKER   = "SPY"
