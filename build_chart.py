#!/usr/bin/env python3
"""Build a self-contained interactive chart of Bitcoin's power-law quantile bands.

Fits 99 quantile regressions of log(Close) against log(days since genesis) and
bakes the resulting coefficients, plus the daily price series, into a single
static HTML file. See docs/superpowers/specs/2026-08-30-interactive-quantile-chart-design.md
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

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


def fit_quantiles(daily: pd.DataFrame) -> np.ndarray:
    """Fit log_Close ~ log_days_since_genesis at each of the 99 quantiles.

    Returns an array of shape (99, 2): column 0 intercept, column 1 slope.
    """
    y = daily["log_Close"]
    x = sm.add_constant(daily["log_days_since_genesis"])

    coef = np.empty((len(QUANTILES), 2), dtype=float)
    for i, q in enumerate(QUANTILES):
        try:
            result = sm.QuantReg(y, x).fit(q=float(q))
        except Exception as exc:
            raise RuntimeError(
                f"QuantReg failed at q={q:.2f}; refusing to emit a partial band"
            ) from exc
        coef[i, 0] = result.params["const"]
        coef[i, 1] = result.params["log_days_since_genesis"]

    if not np.isfinite(coef).all():
        bad = QUANTILES[~np.isfinite(coef).all(axis=1)]
        raise RuntimeError(f"non-finite coefficients at quantiles {bad.tolist()}")
    return coef


@dataclass(frozen=True)
class Crossing:
    """An adjacent quantile pair whose fitted lines invert somewhere in range."""

    lower_q: float
    upper_q: float
    first_day: int


def predict_log_prices(coef: np.ndarray, days) -> np.ndarray:
    """Predicted log price for every quantile at every day.

    Returns shape (len(days), n_quantiles).
    """
    log_days = np.log(np.asarray(days, dtype=float))
    return coef[:, 0][None, :] + coef[:, 1][None, :] * log_days[:, None]


def projection_days(daily: pd.DataFrame, years: int = PROJECTION_YEARS) -> np.ndarray:
    """A log-spaced grid of day values from the first data day to the horizon."""
    first = float(daily["DaysSinceGenesis"].iloc[0])
    last = float(daily["DaysSinceGenesis"].iloc[-1]) + round(years * 365.25)
    return np.exp(np.linspace(np.log(first), np.log(last), 2000))


def check_crossings(
    coef: np.ndarray, quantiles: np.ndarray, days: np.ndarray
) -> list[Crossing]:
    """Find adjacent quantile pairs that invert anywhere in the given day range."""
    preds = predict_log_prices(coef, days)
    inverted = np.diff(preds, axis=1) < 0

    crossings = []
    for j in np.flatnonzero(inverted.any(axis=0)):
        first_day = int(days[int(np.argmax(inverted[:, j]))])
        crossings.append(
            Crossing(
                lower_q=float(quantiles[j]),
                upper_q=float(quantiles[j + 1]),
                first_day=first_day,
            )
        )
    return crossings


def closest_quantile(
    daily: pd.DataFrame, coef: np.ndarray, quantiles: np.ndarray
) -> np.ndarray:
    """The quantile whose fitted line each day's actual close sits nearest to."""
    preds = predict_log_prices(coef, daily["DaysSinceGenesis"].to_numpy())
    actual = daily["log_Close"].to_numpy()[:, None]
    return quantiles[np.abs(preds - actual).argmin(axis=1)]
