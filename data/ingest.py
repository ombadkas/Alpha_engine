"""
Data ingestion pipeline.

Downloads S&P 500 constituents and OHLCV + fundamental data from yfinance,
cleans it, and persists as parquet files ready for factor computation.

Survivorship-bias note: Wikipedia provides *current* constituents only.
Historical constituent data would require a commercial data provider (e.g.
Compustat, CRSP).  This is documented as a known limitation.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from config import DATA_RAW, DATA_PROCESSED, START_DATE, END_DATE, MAX_TICKERS

log = logging.getLogger(__name__)


# ── Universe ───────────────────────────────────────────────────────────────

def get_sp500_tickers() -> list[str]:
    """Scrape current S&P 500 constituents from Wikipedia."""
    import io
    import requests

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        table = pd.read_html(io.StringIO(resp.text), attrs={"id": "constituents"})[0]
        tickers = (
            table["Symbol"]
            .str.replace(".", "-", regex=False)  # BRK.B → BRK-B
            .tolist()
        )
        log.info("Fetched %d tickers from Wikipedia.", len(tickers))
        return tickers if MAX_TICKERS is None else tickers[:MAX_TICKERS]
    except Exception as exc:
        log.error("Failed to fetch S&P 500 tickers: %s", exc)
        raise


# ── OHLCV download ─────────────────────────────────────────────────────────

def download_ohlcv(
    tickers: list[str],
    start: str = START_DATE,
    end: str = END_DATE,
    batch_size: int = 100,
) -> pd.DataFrame:
    """
    Download adjusted-close prices (and volume) for all tickers.

    Returns a MultiIndex DataFrame with columns (field, ticker).
    Batches requests to avoid rate-limiting.
    """
    all_frames: list[pd.DataFrame] = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        log.info("Downloading batch %d–%d …", i + 1, i + len(batch))
        raw = yf.download(
            batch,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        all_frames.append(raw)
        time.sleep(1)  # courteous pause between batches

    data = pd.concat(all_frames, axis=1)
    # Drop duplicate columns that can appear when a ticker appears in two batches
    data = data.loc[:, ~data.columns.duplicated()]
    return data


def extract_close(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Return (date × ticker) adjusted-close price matrix."""
    if isinstance(ohlcv.columns, pd.MultiIndex):
        close = ohlcv["Close"]
    else:
        close = ohlcv[["Close"]]
    return close.sort_index()


# ── Fundamental data ───────────────────────────────────────────────────────

def download_fundamentals(tickers: list[str]) -> pd.DataFrame:
    """
    Pull point-in-time fundamental fields from yfinance .info.

    Fields collected:
        priceToBook   → raw P/B (used for value factor)
        returnOnEquity → trailing ROE (used for quality factor)
        trailingEps    → earnings per share (cross-check)

    Limitation: yfinance .info returns only the most-recent figures.
    For a production system, replace this with a historical fundamentals
    feed (e.g. Compustat via WRDS or Simfin).
    """
    records: list[dict] = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            records.append(
                {
                    "ticker": ticker,
                    "priceToBook": info.get("priceToBook"),
                    "returnOnEquity": info.get("returnOnEquity"),
                    "trailingEps": info.get("trailingEps"),
                    "bookValue": info.get("bookValue"),
                }
            )
        except Exception as exc:
            log.warning("Could not fetch fundamentals for %s: %s", ticker, exc)
            records.append({"ticker": ticker})

    df = pd.DataFrame(records).set_index("ticker")
    log.info(
        "Fundamentals fetched for %d / %d tickers.",
        df.notna().any(axis=1).sum(),
        len(tickers),
    )
    return df


# ── Cleaning ───────────────────────────────────────────────────────────────

def clean_prices(close: pd.DataFrame, min_history: float = 0.8) -> pd.DataFrame:
    """
    Remove tickers with insufficient history, then forward-fill gaps.

    Parameters
    ----------
    close        : (date × ticker) price matrix
    min_history  : minimum fraction of trading days a ticker must have
    """
    threshold = int(len(close) * min_history)
    close = close.dropna(axis=1, thresh=threshold)

    # Drop tickers that are entirely zero or NaN
    close = close.replace(0, np.nan)
    close = close.dropna(axis=1, how="all")

    # Forward-fill within a ticker (up to 5 consecutive missing days)
    close = close.ffill(limit=5)

    # Drop any date rows that are entirely NaN (non-trading days from alignment)
    close = close.dropna(how="all")

    log.info(
        "After cleaning: %d dates × %d tickers.", close.shape[0], close.shape[1]
    )
    return close


# ── Persistence ────────────────────────────────────────────────────────────

def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    log.info("Saved → %s  (%s)", path, df.shape)


def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


# ── Pipeline entry-point ───────────────────────────────────────────────────

def run_ingestion(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full ingestion pipeline.  Returns (close_prices, fundamentals).
    Results are cached as parquet; pass force=True to re-download.
    """
    prices_path = DATA_RAW / "prices.parquet"
    fund_path   = DATA_RAW / "fundamentals.parquet"

    if not force and prices_path.exists() and fund_path.exists():
        log.info("Loading cached data from disk …")
        return load_parquet(prices_path), load_parquet(fund_path)

    tickers = get_sp500_tickers()

    log.info("Downloading OHLCV data …")
    ohlcv   = download_ohlcv(tickers)
    close   = extract_close(ohlcv)
    close   = clean_prices(close)
    save_parquet(close, prices_path)

    log.info("Downloading fundamental data …")
    fundam  = download_fundamentals(close.columns.tolist())
    save_parquet(fundam, fund_path)

    return close, fundam


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    prices, fundamentals = run_ingestion(force=True)
    print(prices.tail())
    print(fundamentals.head())
