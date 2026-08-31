#!/usr/bin/env python3
"""Build a self-contained interactive chart of Bitcoin's power-law quantile bands.

Fits 99 quantile regressions of log(Close) against log(days since genesis) and
bakes the resulting coefficients, plus the daily price series, into a single
static HTML file. See docs/superpowers/specs/2026-08-30-interactive-quantile-chart-design.md
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
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


def build_payload(
    daily: pd.DataFrame,
    coef: np.ndarray,
    quantiles: np.ndarray,
    closest: np.ndarray,
    crossings: list[Crossing],
) -> dict:
    """The complete data the browser needs, ready for json.dumps."""
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "genesis": GENESIS_DATE.strftime("%Y-%m-%d"),
        "last_date": pd.Timestamp(daily["Date"].iloc[-1]).strftime("%Y-%m-%d"),
        "last_close": round(float(daily["Close"].iloc[-1]), 2),
        "projection_years": PROJECTION_YEARS,
        "quantiles": [round(float(q), 2) for q in quantiles],
        "coef": [[float(a), float(b)] for a, b in coef],
        "days": [int(d) for d in daily["DaysSinceGenesis"]],
        "close": [round(float(c), 2) for c in daily["Close"]],
        "closest_q": [round(float(q), 2) for q in closest],
        "crossings_in_range": len(crossings) > 0,
    }


def render(payload: dict, template_path: Path, out_path: Path) -> None:
    """Substitute the payload into the template and write the standalone HTML."""
    template = Path(template_path).read_text(encoding="utf-8")
    if TEMPLATE_MARKER not in template:
        raise ValueError(f"template {template_path} is missing marker {TEMPLATE_MARKER}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        template.replace(TEMPLATE_MARKER, json.dumps(payload, separators=(",", ":"))),
        encoding="utf-8",
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="minute-level CSV to use instead of downloading from Kaggle",
    )
    parser.add_argument(
        "--template", type=Path, default=Path(__file__).parent / "chart_template.html"
    )
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).parent / "docs" / "index.html"
    )
    args = parser.parse_args(argv)

    csv_path = args.csv if args.csv is not None else download_minute_csv()
    print(f"Reading {csv_path}")
    daily = to_daily(pd.read_csv(csv_path))
    print(
        f"{len(daily):,} daily rows from {daily['Date'].iloc[0]:%Y-%m-%d} "
        f"to {daily['Date'].iloc[-1]:%Y-%m-%d} ({daily.attrs['dropped']} dropped)"
    )

    print(f"Fitting {len(QUANTILES)} quantile regressions...")
    coef = fit_quantiles(daily)

    median = sm.QuantReg(
        daily["log_Close"], sm.add_constant(daily["log_days_since_genesis"])
    ).fit(q=0.5)
    print(f"Median-quantile pseudo R-squared: {median.prsquared:.4f}")

    days_grid = projection_days(daily)
    crossings = check_crossings(coef, QUANTILES, days_grid)
    if crossings:
        print(f"WARNING: {len(crossings)} adjacent quantile pair(s) cross in range:")
        for c in crossings[:10]:
            date = GENESIS_DATE + pd.Timedelta(days=c.first_day)
            print(f"  q{c.lower_q:.2f}/q{c.upper_q:.2f} from {date:%Y-%m-%d}")
        print("  The chart sorts quantile values at evaluation time to compensate.")

    closest = closest_quantile(daily, coef, QUANTILES)
    payload = build_payload(daily, coef, QUANTILES, closest, crossings)
    render(payload, args.template, args.out)

    size_kb = args.out.stat().st_size / 1024
    print(f"Wrote {args.out} ({size_kb:,.0f} KB)")
    print(f"Latest close {payload['last_close']:,.2f} at q{payload['closest_q'][-1]:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

