#!/usr/bin/env python3
"""Build a self-contained interactive chart of Bitcoin's power-law quantile bands.

Fits 99 quantile regressions of log(Close) against log(days since genesis) and
bakes the resulting coefficients, plus the daily price series, into a single
static HTML file. See docs/superpowers/specs/2026-08-30-interactive-quantile-chart-design.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# The notebook uses 2010-01-03 for DaysSinceGenesis (cell 14), while its halving
# list uses 2009-01-09 and bitcoin_quantile_fixes.py uses 2009-01-03. We keep the
# notebook's value so this chart's numbers match the existing analysis.
GENESIS_DATE = pd.Timestamp("2010-01-03")

QUANTILES = np.round(np.arange(1, 100) / 100.0, 2)

KAGGLE_DATASET = "mczielinski/bitcoin-historical-data"
MINUTE_CSV = "btcusd_1-min_data.csv"
TEMPLATE_MARKER = "/*__PAYLOAD__*/"
PROJECTION_YEARS = 10


def download_minute_csv() -> Path:
    """Download the Kaggle dataset and return the path to the minute-level CSV."""
    import kagglehub

    try:
        path = kagglehub.dataset_download(KAGGLE_DATASET)
    except Exception as exc:
        raise RuntimeError(
            f"Kaggle download failed for {KAGGLE_DATASET}: {exc}. "
            "kagglehub reuses its cache under ~/.cache/kagglehub when it can; "
            "check credentials in ~/.kaggle/kaggle.json."
        ) from exc

    csv_path = Path(path) / MINUTE_CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"{MINUTE_CSV} not found in downloaded dataset at {path}")
    return csv_path


def to_daily(minute_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate minute-level OHLCV to daily and add the power-law columns."""
    df = minute_df.copy()
    df["Date"] = pd.to_datetime(df["Timestamp"], unit="s").dt.floor("D")
    df = df.sort_values("Timestamp")

    daily = df.groupby("Date", as_index=False).agg(
        Open=("Open", "first"),
        High=("High", "max"),
        Low=("Low", "min"),
        Close=("Close", "last"),
        Volume=("Volume", "sum"),
    )
    daily["DaysSinceGenesis"] = (daily["Date"] - GENESIS_DATE).dt.days

    before = len(daily)
    keep = (daily["Close"] > 0) & (daily["DaysSinceGenesis"] > 0) & daily["Close"].notna()
    daily = daily[keep].reset_index(drop=True)

    daily["log_Close"] = np.log(daily["Close"])
    daily["log_days_since_genesis"] = np.log(daily["DaysSinceGenesis"])
    daily.attrs["dropped"] = before - len(daily)
    return daily
