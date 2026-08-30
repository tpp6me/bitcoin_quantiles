# Interactive Power-Law Quantile Chart Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained HTML chart of Bitcoin's 99 fitted power-law quantile bands, where hovering any date — historical or projected — shows the price at every quantile.

**Architecture:** A Python build script (`build_chart.py`) downloads Kaggle minute data, aggregates to daily, fits 99 quantile regressions of `log_Close ~ log_days_since_genesis`, and bakes the resulting 99 intercept/slope pairs plus the daily price series into a JSON payload. That payload is substituted into `chart_template.html` at a marker, producing `docs/index.html`. Because each quantile is the closed form `exp(a_q + b_q·ln(days))`, the browser evaluates all 99 quantiles for any date exactly, in JavaScript, with no interpolation and no model.

**Tech Stack:** Python 3.12, pandas, numpy, statsmodels (`QuantReg`), kagglehub, pytest. Browser side: Plotly.js 2.35.2 from cdnjs, vanilla JS, no build step.

**Spec:** `docs/superpowers/specs/2026-08-30-interactive-quantile-chart-design.md`

## Global Constraints

- Genesis date is **2010-01-03** — matches notebook cell 14. One constant, `GENESIS_DATE`. Do not use 2009-01-03 or 2009-01-09.
- Exactly **99 quantiles**: 0.01 through 0.99 in steps of 0.01, rounded to 2 decimal places.
- Output is **`docs/index.html`**, committed to the repo, and must open correctly over `file://` with no local server.
- The only permitted network reference in the output is the pinned Plotly CDN URL: `https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.2/plotly.min.js`. Nothing else is fetched at runtime.
- Template substitution marker is the literal string `/*__PAYLOAD__*/`.
- Projection horizon used when building traces is always **10 years** past the last data point; the horizon control only changes the visible axis range.
- Quantile rearrangement (sorting the 99 evaluated prices ascending) happens **in the browser at evaluation time**. Emitted coefficients are never modified.
- Only one new Python dependency: `pytest`. No Plotly Python, no Jinja.
- All work happens inside the `bitcoin_quantiles/` repo (its own git repo, remote `tpp6me/bitcoin_quantiles`).

---

## File Structure

| File | Responsibility |
|---|---|
| `build_chart.py` | Constants, data loading, quantile fitting, crossing detection, payload assembly, template rendering, CLI. |
| `chart_template.html` | The entire browser UI — markup, CSS, JS. One `/*__PAYLOAD__*/` marker. |
| `docs/index.html` | Generated output. Committed. |
| `tests/test_build_chart.py` | Unit tests for every pure function in `build_chart.py`. |
| `tests/test_chart_html.py` | Structural tests on the template and the rendered output. |
| `requirements.txt` | Modified: add `pytest`. |

`build_chart.py` stays a single module. It is under ~250 lines of small pure functions plus a thin `main()`, and the repo's existing convention is flat top-level scripts (`bitcoin_quantile_fixes.py`).

---

### Task 1: Constants, data loading, daily aggregation

**Files:**
- Create: `build_chart.py`
- Create: `tests/test_build_chart.py`
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `GENESIS_DATE: pd.Timestamp`, `QUANTILES: np.ndarray` (shape `(99,)`), `KAGGLE_DATASET: str`, `MINUTE_CSV: str`, `TEMPLATE_MARKER: str`, `PROJECTION_YEARS: int`
  - `download_minute_csv() -> pathlib.Path`
  - `to_daily(minute_df: pd.DataFrame) -> pd.DataFrame` — returns columns `Date` (datetime64), `Open`, `High`, `Low`, `Close`, `Volume`, `DaysSinceGenesis` (int), `log_Close`, `log_days_since_genesis`; sets `daily.attrs["dropped"]` to the number of rows removed.

- [ ] **Step 1: Add pytest to requirements**

Append one line to `requirements.txt` (the file is otherwise a `pip freeze` dump; it will be re-pinned on the next freeze):

```
pytest
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_build_chart.py`:

```python
import numpy as np
import pandas as pd
import pytest

from build_chart import GENESIS_DATE, QUANTILES, to_daily


def test_quantiles_are_ninety_nine_two_decimal_values():
    assert QUANTILES.shape == (99,)
    assert QUANTILES[0] == pytest.approx(0.01)
    assert QUANTILES[-1] == pytest.approx(0.99)
    assert np.allclose(np.diff(QUANTILES), 0.01)


def test_genesis_date_matches_notebook():
    assert GENESIS_DATE == pd.Timestamp("2010-01-03")


def _minute_frame(rows):
    """rows: list of (iso_datetime, open, high, low, close, volume)"""
    ts = pd.to_datetime([r[0] for r in rows], utc=True)
    return pd.DataFrame(
        {
            "Timestamp": ts.astype("int64") // 10**9,
            "Open": [r[1] for r in rows],
            "High": [r[2] for r in rows],
            "Low": [r[3] for r in rows],
            "Close": [r[4] for r in rows],
            "Volume": [r[5] for r in rows],
        }
    )


def test_to_daily_aggregates_minutes_into_days():
    minute_df = _minute_frame(
        [
            ("2015-01-01 00:00", 10.0, 11.0, 9.0, 10.5, 1.0),
            ("2015-01-01 23:59", 12.0, 15.0, 11.0, 14.0, 2.0),
            ("2015-01-02 12:00", 20.0, 25.0, 19.0, 24.0, 3.0),
        ]
    )
    daily = to_daily(minute_df)

    assert len(daily) == 2
    assert daily.loc[0, "Open"] == 10.0
    assert daily.loc[0, "High"] == 15.0
    assert daily.loc[0, "Low"] == 9.0
    assert daily.loc[0, "Close"] == 14.0
    assert daily.loc[0, "Volume"] == 3.0


def test_to_daily_derives_days_and_logs_from_genesis():
    minute_df = _minute_frame([("2015-01-01 00:00", 10.0, 11.0, 9.0, 10.0, 1.0)])
    daily = to_daily(minute_df)

    expected_days = (pd.Timestamp("2015-01-01") - GENESIS_DATE).days
    assert daily.loc[0, "DaysSinceGenesis"] == expected_days
    assert daily.loc[0, "log_Close"] == pytest.approx(np.log(10.0))
    assert daily.loc[0, "log_days_since_genesis"] == pytest.approx(np.log(expected_days))


def test_to_daily_drops_nonpositive_prices_and_records_the_count():
    minute_df = _minute_frame(
        [
            ("2015-01-01 00:00", 10.0, 11.0, 9.0, 0.0, 1.0),
            ("2015-01-02 00:00", 10.0, 11.0, 9.0, 10.0, 1.0),
        ]
    )
    daily = to_daily(minute_df)

    assert len(daily) == 1
    assert daily.loc[0, "Close"] == 10.0
    assert daily.attrs["dropped"] == 1
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_build_chart.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'build_chart'`

- [ ] **Step 4: Write the implementation**

Create `build_chart.py`:

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_build_chart.py -v`
Expected: PASS, 5 tests

- [ ] **Step 6: Commit**

```bash
git add build_chart.py tests/test_build_chart.py requirements.txt
git commit -m "feat: add daily aggregation and power-law constants for chart build"
```

---

### Task 2: Fit the 99 quantile regressions

**Files:**
- Modify: `build_chart.py`
- Modify: `tests/test_build_chart.py`

**Interfaces:**
- Consumes: `QUANTILES`, and a daily frame with `log_Close` and `log_days_since_genesis` columns from `to_daily`.
- Produces: `fit_quantiles(daily: pd.DataFrame) -> np.ndarray` of shape `(99, 2)`, column 0 = intercept, column 1 = slope, rows in ascending quantile order.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_build_chart.py`:

```python
from build_chart import fit_quantiles


def _synthetic_daily(intercept=2.0, slope=1.5, sigma=0.2, seed=0):
    days = np.arange(500, 5000)
    rng = np.random.default_rng(seed)
    log_close = intercept + slope * np.log(days) + rng.normal(0.0, sigma, len(days))
    return pd.DataFrame(
        {
            "DaysSinceGenesis": days,
            "log_days_since_genesis": np.log(days),
            "log_Close": log_close,
            "Close": np.exp(log_close),
        }
    )


def test_fit_quantiles_returns_one_pair_per_quantile():
    coef = fit_quantiles(_synthetic_daily())
    assert coef.shape == (99, 2)
    assert np.isfinite(coef).all()


def test_fit_quantiles_recovers_the_median_line():
    coef = fit_quantiles(_synthetic_daily(intercept=2.0, slope=1.5, sigma=0.2))
    median_row = 49  # QUANTILES[49] == 0.50

    assert coef[median_row, 1] == pytest.approx(1.5, abs=0.05)

    at_2000 = coef[median_row, 0] + coef[median_row, 1] * np.log(2000)
    assert at_2000 == pytest.approx(2.0 + 1.5 * np.log(2000), abs=0.03)


def test_fit_quantiles_spreads_by_the_noise_distribution():
    sigma = 0.2
    coef = fit_quantiles(_synthetic_daily(sigma=sigma))
    log_2000 = np.log(2000)

    def predict(row):
        return coef[row, 0] + coef[row, 1] * log_2000

    # q0.90 sits about 1.2816 standard deviations above the median.
    assert predict(89) - predict(49) == pytest.approx(1.2816 * sigma, abs=0.05)
    assert predict(49) - predict(9) == pytest.approx(1.2816 * sigma, abs=0.05)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_build_chart.py -k fit_quantiles -v`
Expected: FAIL with `ImportError: cannot import name 'fit_quantiles'`

- [ ] **Step 3: Write the implementation**

Add to `build_chart.py` (import at the top with the others):

```python
import statsmodels.api as sm
```

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_build_chart.py -v`
Expected: PASS, 8 tests

- [ ] **Step 5: Commit**

```bash
git add build_chart.py tests/test_build_chart.py
git commit -m "feat: fit 99 quantile regressions for the power-law band"
```

---

### Task 3: Predict prices and detect quantile crossings

**Files:**
- Modify: `build_chart.py`
- Modify: `tests/test_build_chart.py`

**Interfaces:**
- Consumes: `fit_quantiles` output, `QUANTILES`.
- Produces:
  - `predict_log_prices(coef: np.ndarray, days) -> np.ndarray` of shape `(len(days), 99)`
  - `Crossing` — a frozen dataclass with fields `lower_q: float`, `upper_q: float`, `first_day: int`
  - `check_crossings(coef: np.ndarray, quantiles: np.ndarray, days: np.ndarray) -> list[Crossing]`
  - `projection_days(daily: pd.DataFrame, years: int = PROJECTION_YEARS) -> np.ndarray` — a log-spaced grid of 2000 day values from the first data day to `last_day + round(years * 365.25)`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_build_chart.py`:

```python
from build_chart import Crossing, check_crossings, predict_log_prices, projection_days


def test_predict_log_prices_shape_and_values():
    coef = np.array([[0.0, 1.0], [1.0, 2.0]])
    days = np.array([10.0, 100.0])
    preds = predict_log_prices(coef, days)

    assert preds.shape == (2, 2)
    assert preds[0, 0] == pytest.approx(np.log(10.0))
    assert preds[1, 1] == pytest.approx(1.0 + 2.0 * np.log(100.0))


def test_check_crossings_finds_an_inversion():
    # Lines cross where ln(d) == 2, i.e. d ~= 7.39.
    coef = np.array([[0.0, 1.0], [1.0, 0.5]])
    quantiles = np.array([0.50, 0.60])
    days = np.arange(2, 101)

    crossings = check_crossings(coef, quantiles, days)

    assert len(crossings) == 1
    assert crossings[0] == Crossing(lower_q=0.50, upper_q=0.60, first_day=8)


def test_check_crossings_returns_empty_for_parallel_lines():
    coef = np.array([[0.0, 1.0], [1.0, 1.0]])
    quantiles = np.array([0.50, 0.60])
    days = np.arange(2, 10001)

    assert check_crossings(coef, quantiles, days) == []


def test_projection_days_spans_data_start_to_the_horizon():
    daily = pd.DataFrame({"DaysSinceGenesis": np.arange(1000, 2001)})
    days = projection_days(daily, years=10)

    assert len(days) == 2000
    assert days[0] == pytest.approx(1000.0)
    assert days[-1] == pytest.approx(2000 + round(10 * 365.25))
    assert np.all(np.diff(days) > 0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_build_chart.py -k "predict or crossings or projection" -v`
Expected: FAIL with `ImportError: cannot import name 'Crossing'`

- [ ] **Step 3: Write the implementation**

Add to `build_chart.py` (import `from dataclasses import dataclass` at the top):

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_build_chart.py -v`
Expected: PASS, 12 tests

- [ ] **Step 5: Commit**

```bash
git add build_chart.py tests/test_build_chart.py
git commit -m "feat: predict quantile prices and detect crossings across the horizon"
```

---

### Task 4: Map each day to its closest quantile

**Files:**
- Modify: `build_chart.py`
- Modify: `tests/test_build_chart.py`

**Interfaces:**
- Consumes: `predict_log_prices`, a daily frame with `DaysSinceGenesis` and `log_Close`.
- Produces: `closest_quantile(daily: pd.DataFrame, coef: np.ndarray, quantiles: np.ndarray) -> np.ndarray` of length `len(daily)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_build_chart.py`:

```python
from build_chart import closest_quantile


def test_closest_quantile_picks_the_line_the_point_sits_on():
    # Three parallel lines, one unit apart, tagged 0.25 / 0.50 / 0.75.
    coef = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
    quantiles = np.array([0.25, 0.50, 0.75])
    days = np.array([100.0, 100.0, 100.0])
    log_days = np.log(days)

    daily = pd.DataFrame(
        {
            "DaysSinceGenesis": days,
            # exactly on the 0.25 line, the 0.50 line, and the 0.75 line
            "log_Close": [log_days[0], 1.0 + log_days[1], 2.0 + log_days[2]],
        }
    )

    result = closest_quantile(daily, coef, quantiles)

    assert result.tolist() == [0.25, 0.50, 0.75]


def test_closest_quantile_returns_one_value_per_row():
    daily = _synthetic_daily()
    coef = fit_quantiles(daily)
    result = closest_quantile(daily, coef, QUANTILES)

    assert len(result) == len(daily)
    assert set(result).issubset(set(QUANTILES.tolist()))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_build_chart.py -k closest_quantile -v`
Expected: FAIL with `ImportError: cannot import name 'closest_quantile'`

- [ ] **Step 3: Write the implementation**

Add to `build_chart.py`:

```python
def closest_quantile(
    daily: pd.DataFrame, coef: np.ndarray, quantiles: np.ndarray
) -> np.ndarray:
    """The quantile whose fitted line each day's actual close sits nearest to."""
    preds = predict_log_prices(coef, daily["DaysSinceGenesis"].to_numpy())
    actual = daily["log_Close"].to_numpy()[:, None]
    return quantiles[np.abs(preds - actual).argmin(axis=1)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_build_chart.py -v`
Expected: PASS, 14 tests

- [ ] **Step 5: Commit**

```bash
git add build_chart.py tests/test_build_chart.py
git commit -m "feat: map each daily close to its closest fitted quantile"
```

---

### Task 5: Assemble the JSON payload

**Files:**
- Modify: `build_chart.py`
- Modify: `tests/test_build_chart.py`

**Interfaces:**
- Consumes: everything from Tasks 1-4.
- Produces: `build_payload(daily, coef, quantiles, closest, crossings) -> dict` with exactly these keys: `generated_at`, `genesis`, `last_date`, `last_close`, `projection_years`, `quantiles`, `coef`, `days`, `close`, `closest_q`, `crossings_in_range`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_build_chart.py`:

```python
import json

from build_chart import build_payload


def _payload_fixture():
    daily = _synthetic_daily()
    daily["Date"] = pd.date_range("2011-05-17", periods=len(daily), freq="D")
    coef = fit_quantiles(daily)
    closest = closest_quantile(daily, coef, QUANTILES)
    crossings = check_crossings(coef, QUANTILES, projection_days(daily))
    return build_payload(daily, coef, QUANTILES, closest, crossings), daily


def test_build_payload_has_the_expected_keys():
    payload, _ = _payload_fixture()

    assert set(payload) == {
        "generated_at",
        "genesis",
        "last_date",
        "last_close",
        "projection_years",
        "quantiles",
        "coef",
        "days",
        "close",
        "closest_q",
        "crossings_in_range",
    }


def test_build_payload_carries_ninety_nine_quantiles_and_coefficients():
    payload, _ = _payload_fixture()

    assert len(payload["quantiles"]) == 99
    assert len(payload["coef"]) == 99
    assert all(len(pair) == 2 for pair in payload["coef"])
    assert payload["genesis"] == "2010-01-03"
    assert payload["projection_years"] == 10


def test_build_payload_series_arrays_are_parallel_and_clean():
    payload, daily = _payload_fixture()

    n = len(daily)
    assert len(payload["days"]) == n
    assert len(payload["close"]) == n
    assert len(payload["closest_q"]) == n
    assert all(isinstance(d, int) for d in payload["days"])
    assert payload["last_close"] == pytest.approx(round(float(daily["Close"].iloc[-1]), 2))


def test_build_payload_is_json_serialisable_and_finite():
    payload, _ = _payload_fixture()
    text = json.dumps(payload)
    reloaded = json.loads(text)

    assert "NaN" not in text
    assert "Infinity" not in text
    assert np.isfinite(np.array(reloaded["coef"])).all()
    assert np.isfinite(np.array(reloaded["close"])).all()


def test_build_payload_flags_crossings():
    daily = _synthetic_daily()
    daily["Date"] = pd.date_range("2011-05-17", periods=len(daily), freq="D")
    coef = fit_quantiles(daily)
    closest = closest_quantile(daily, coef, QUANTILES)
    fake = [Crossing(lower_q=0.90, upper_q=0.91, first_day=9000)]

    payload = build_payload(daily, coef, QUANTILES, closest, fake)

    assert payload["crossings_in_range"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_build_chart.py -k build_payload -v`
Expected: FAIL with `ImportError: cannot import name 'build_payload'`

- [ ] **Step 3: Write the implementation**

Add to `build_chart.py` (import `from datetime import datetime, timezone` at the top):

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_build_chart.py -v`
Expected: PASS, 19 tests

- [ ] **Step 5: Commit**

```bash
git add build_chart.py tests/test_build_chart.py
git commit -m "feat: assemble the chart JSON payload"
```

---

### Task 6: Template skeleton and rendering

**Files:**
- Create: `chart_template.html`
- Create: `tests/test_chart_html.py`
- Modify: `build_chart.py`

**Interfaces:**
- Consumes: `TEMPLATE_MARKER`, `build_payload` output.
- Produces: `render(payload: dict, template_path: Path, out_path: Path) -> None`

- [ ] **Step 1: Write the failing test**

Create `tests/test_chart_html.py`:

```python
import json
from pathlib import Path

import pytest

from build_chart import TEMPLATE_MARKER, render

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPO_ROOT / "chart_template.html"
PLOTLY_CDN = "https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.2/plotly.min.js"


def test_template_exists_and_carries_the_marker():
    text = TEMPLATE.read_text(encoding="utf-8")
    assert TEMPLATE_MARKER in text
    assert text.count(TEMPLATE_MARKER) == 1


def test_template_pins_plotly_and_references_nothing_else_remote():
    text = TEMPLATE.read_text(encoding="utf-8")
    assert PLOTLY_CDN in text
    remote = [
        line
        for line in text.splitlines()
        if ("http://" in line or "https://" in line) and PLOTLY_CDN not in line
    ]
    assert remote == [], f"unexpected remote references: {remote}"


def test_render_substitutes_the_payload(tmp_path):
    payload = {"hello": "world", "n": 1}
    out = tmp_path / "out" / "index.html"

    render(payload, TEMPLATE, out)

    text = out.read_text(encoding="utf-8")
    assert TEMPLATE_MARKER not in text
    assert json.dumps(payload, separators=(",", ":")) in text


def test_render_rejects_a_template_without_the_marker(tmp_path):
    bad = tmp_path / "bad.html"
    bad.write_text("<html></html>", encoding="utf-8")

    with pytest.raises(ValueError, match="marker"):
        render({}, bad, tmp_path / "out.html")


def test_rendered_output_has_no_local_file_references(tmp_path):
    out = tmp_path / "index.html"
    render({"quantiles": []}, TEMPLATE, out)
    text = out.read_text(encoding="utf-8")

    assert 'src="./' not in text
    assert 'href="./' not in text
    assert 'src="/' not in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_chart_html.py -v`
Expected: FAIL with `ImportError: cannot import name 'render'` and a missing `chart_template.html`

- [ ] **Step 3: Create the template skeleton**

Create `chart_template.html`:

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bitcoin Power-Law Quantiles</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.2/plotly.min.js"></script>
<style>
  :root {
    --bg: #0e1116;
    --panel: #161b22;
    --line: #2a313c;
    --text: #c9d1d9;
    --dim: #7d8590;
    --accent: #f5b301;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--text);
    font: 13px/1.45 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  header {
    display: flex; flex-wrap: wrap; gap: 16px; align-items: baseline;
    padding: 14px 18px; border-bottom: 1px solid var(--line);
  }
  h1 { font-size: 15px; font-weight: 600; margin: 0; letter-spacing: .2px; }
  .meta { color: var(--dim); font-size: 12px; }
  .controls { display: flex; gap: 18px; margin-left: auto; flex-wrap: wrap; }
  .control { display: flex; align-items: center; gap: 6px; }
  .control label { color: var(--dim); font-size: 11px; text-transform: uppercase; letter-spacing: .6px; }
  .seg { display: flex; border: 1px solid var(--line); border-radius: 5px; overflow: hidden; }
  .seg button {
    background: transparent; color: var(--dim); border: 0; padding: 4px 10px;
    font: inherit; font-size: 12px; cursor: pointer;
  }
  .seg button[aria-pressed="true"] { background: var(--line); color: var(--text); }
  main { display: flex; align-items: stretch; height: calc(100vh - 56px); }
  #chart { flex: 1 1 auto; min-width: 0; }
  #panel {
    flex: 0 0 300px; border-left: 1px solid var(--line); background: var(--panel);
    padding: 14px 16px; overflow-y: auto;
  }
  #panel h2 { font-size: 13px; margin: 0 0 2px; font-weight: 600; }
  #panel .sub { color: var(--dim); font-size: 12px; margin-bottom: 12px; }
  #panel .actual {
    border: 1px solid var(--line); border-radius: 6px; padding: 8px 10px; margin-bottom: 12px;
  }
  #panel .actual .price { font-size: 18px; font-weight: 600; }
  #panel .actual .q { color: var(--accent); font-size: 12px; }
  table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
  td { padding: 2px 0; }
  td.q { color: var(--dim); width: 52px; }
  td.p { text-align: right; }
  tr.bracket { background: rgba(245, 179, 1, .10); }
  tr.bracket td.q { color: var(--accent); }
  .hint { color: var(--dim); font-size: 11px; margin-top: 14px; }
  @media (max-width: 860px) {
    main { flex-direction: column; height: auto; }
    #chart { height: 60vh; }
    #panel { flex: 1 1 auto; border-left: 0; border-top: 1px solid var(--line); }
  }
</style>
</head>
<body>
<header>
  <div>
    <h1>Bitcoin Power-Law Quantiles</h1>
    <div class="meta" id="meta"></div>
  </div>
  <div class="controls">
    <div class="control">
      <label for="horizon">Horizon</label>
      <div class="seg" id="horizon">
        <button data-years="0" aria-pressed="false">Today</button>
        <button data-years="2" aria-pressed="false">+2y</button>
        <button data-years="5" aria-pressed="true">+5y</button>
        <button data-years="10" aria-pressed="false">+10y</button>
      </div>
    </div>
    <div class="control">
      <label for="density">Readout</label>
      <div class="seg" id="density">
        <button data-step="10" aria-pressed="false">11</button>
        <button data-step="5" aria-pressed="true">21</button>
        <button data-step="1" aria-pressed="false">All 99</button>
      </div>
    </div>
  </div>
</header>
<main>
  <div id="chart"></div>
  <aside id="panel">
    <h2 id="panel-date">Hover the chart</h2>
    <div class="sub" id="panel-sub">Prices at every quantile for that date.</div>
    <div class="actual" id="panel-actual" hidden></div>
    <table><tbody id="panel-rows"></tbody></table>
    <div class="hint">Click to pin the readout. Click again to release.</div>
  </aside>
</main>
<script>
const PAYLOAD = /*__PAYLOAD__*/;
</script>
</body>
</html>
```

- [ ] **Step 4: Write `render`**

Add to `build_chart.py` (import `import json` at the top):

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/ -v`
Expected: PASS, 24 tests

- [ ] **Step 6: Commit**

```bash
git add chart_template.html tests/test_chart_html.py build_chart.py
git commit -m "feat: add chart template skeleton and payload rendering"
```

---

### Task 7: Draw the band, emphasis lines, and price

**Files:**
- Modify: `chart_template.html`
- Modify: `tests/test_chart_html.py`

**Interfaces:**
- Consumes: `PAYLOAD` in the rendered page.
- Produces: browser globals `window.__quantilePricesAt(days) -> number[]` (99 prices, ascending) and `window.__chartReady` (a Promise resolving once Plotly has drawn), both relied on by Task 10's verification.

- [ ] **Step 1: Write the failing structural test**

Append to `tests/test_chart_html.py`:

```python
def test_template_exposes_the_browser_hooks_task_ten_verifies():
    text = TEMPLATE.read_text(encoding="utf-8")
    for hook in ("__quantilePricesAt", "__chartReady"):
        assert f"window.{hook}" in text, f"missing browser hook {hook}"


def test_template_builds_all_ninety_eight_bands():
    text = TEMPLATE.read_text(encoding="utf-8")
    assert "tonexty" in text
    assert "Plotly.newPlot" in text
    assert "EMPHASIS" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_chart_html.py -k "hooks or bands" -v`
Expected: FAIL — `missing browser hook __quantilePricesAt`

- [ ] **Step 3: Add the chart script**

In `chart_template.html`, replace the final `<script>` block with:

```html
<script>
const PAYLOAD = /*__PAYLOAD__*/;

const P = PAYLOAD;
const NQ = P.quantiles.length;
const DAY_MS = 86400000;
const GENESIS_MS = Date.parse(P.genesis + "T00:00:00Z");
const EMPHASIS = [0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99];

const FIRST_DAY = P.days[0];
const LAST_DAY = P.days[P.days.length - 1];
const MAX_DAY = LAST_DAY + Math.round(P.projection_years * 365.25);

const gd = document.getElementById("chart");

function daysToDate(d) { return new Date(GENESIS_MS + d * DAY_MS); }
function fmtDate(d) { return daysToDate(d).toISOString().slice(0, 10); }

function fmtPrice(v) {
  if (v >= 1e9) return "$" + (v / 1e9).toFixed(2) + "B";
  if (v >= 1e6) return "$" + (v / 1e6).toFixed(2) + "M";
  if (v >= 1e3) return "$" + (v / 1e3).toFixed(1) + "k";
  if (v >= 1) return "$" + v.toFixed(2);
  return "$" + v.toPrecision(3);
}

// The whole model: 99 closed-form lines, rearranged (sorted) at evaluation time
// so a higher quantile can never report a lower price. See the spec's
// "Quantile crossing" section.
function quantilePricesAt(days) {
  const ld = Math.log(days);
  const out = new Array(NQ);
  for (let i = 0; i < NQ; i++) out[i] = Math.exp(P.coef[i][0] + P.coef[i][1] * ld);
  out.sort((a, b) => a - b);
  return out;
}
window.__quantilePricesAt = quantilePricesAt;

function qIndex(q) {
  let best = 0, bestDiff = Infinity;
  for (let i = 0; i < NQ; i++) {
    const d = Math.abs(P.quantiles[i] - q);
    if (d < bestDiff) { bestDiff = d; best = i; }
  }
  return best;
}

function logspace(a, b, n) {
  const la = Math.log(a), lb = Math.log(b), out = new Array(n);
  for (let k = 0; k < n; k++) out[k] = Math.exp(la + (lb - la) * k / (n - 1));
  return out;
}

// Straight lines in log-log space need only their endpoints. When the fit has
// crossings inside the range, sorting makes each band piecewise, so sample it.
const BAND_X = P.crossings_in_range
  ? logspace(FIRST_DAY, MAX_DAY, 200)
  : [FIRST_DAY, MAX_DAY];

const BAND_Y = (function () {
  const series = Array.from({ length: NQ }, () => new Array(BAND_X.length));
  BAND_X.forEach((d, k) => {
    const prices = quantilePricesAt(d);
    for (let i = 0; i < NQ; i++) series[i][k] = prices[i];
  });
  return series;
})();

function bandColor(t) {
  return "hsla(" + Math.round(212 - 212 * t) + ", 72%, 54%, 0.5)";
}

function yearTicks() {
  const vals = [], text = [];
  const first = daysToDate(FIRST_DAY).getUTCFullYear();
  const last = daysToDate(MAX_DAY).getUTCFullYear();
  let lastKept = -Infinity;
  for (let y = first; y <= last; y++) {
    const d = (Date.UTC(y, 0, 1) - GENESIS_MS) / DAY_MS;
    if (d <= FIRST_DAY || d > MAX_DAY) continue;
    const l = Math.log10(d);
    // Drop labels that would collide on a log axis.
    if (l - lastKept < 0.045 && y !== last) continue;
    lastKept = l;
    vals.push(d);
    text.push(String(y));
  }
  return { vals, text };
}

function buildTraces() {
  const traces = [];

  for (let i = 0; i < NQ; i++) {
    traces.push({
      x: BAND_X, y: BAND_Y[i], type: "scatter", mode: "lines",
      line: { width: 0 }, hoverinfo: "skip", showlegend: false,
      fill: i === 0 ? "none" : "tonexty",
      fillcolor: bandColor(i / (NQ - 1)),
    });
  }

  EMPHASIS.forEach(function (q) {
    const i = qIndex(q);
    traces.push({
      x: BAND_X, y: BAND_Y[i], type: "scatter", mode: "lines",
      line: {
        width: q === 0.5 ? 1.8 : 1.1,
        color: q === 0.5 ? "rgba(255,255,255,.92)" : "rgba(255,255,255,.55)",
        dash: q === 0.5 ? "solid" : "dot",
      },
      hoverinfo: "skip", showlegend: false,
    });
  });

  traces.push({
    x: P.days, y: P.close, type: "scatter", mode: "lines",
    line: { width: 1.1, color: "#0b0d11" },
    hoverinfo: "skip", showlegend: false,
  });

  // Invisible dense grid so plotly_hover fires anywhere, projection included.
  const hoverX = logspace(FIRST_DAY, MAX_DAY, 800);
  const medianRow = qIndex(0.5);
  traces.push({
    x: hoverX,
    y: hoverX.map(function (d) { return quantilePricesAt(d)[medianRow]; }),
    type: "scatter", mode: "markers",
    marker: { opacity: 0, size: 1 },
    hoverinfo: "none", showlegend: false, name: "hovercatch",
  });

  return traces;
}

function emphasisAnnotations(maxDay) {
  return EMPHASIS.map(function (q) {
    const i = qIndex(q);
    return {
      x: Math.log10(maxDay), y: Math.log10(quantilePricesAt(maxDay)[i]),
      xref: "x", yref: "y", text: "q" + q.toFixed(2),
      xanchor: "left", xshift: 4, showarrow: false,
      font: { size: 10, color: "rgba(255,255,255,.65)" },
    };
  });
}

function axisRanges(years) {
  const d1 = years === 0 ? LAST_DAY : LAST_DAY + Math.round(years * 365.25);
  const lo = quantilePricesAt(FIRST_DAY)[0] * 0.5;
  const hi = quantilePricesAt(d1)[NQ - 1] * 1.6;
  return {
    x: [Math.log10(FIRST_DAY), Math.log10(d1)],
    y: [Math.log10(lo), Math.log10(hi)],
    maxDay: d1,
  };
}

const ticks = yearTicks();
const initial = axisRanges(5);

const layout = {
  paper_bgcolor: "#0e1116", plot_bgcolor: "#0e1116",
  font: { color: "#c9d1d9", family: "ui-sans-serif, system-ui, sans-serif", size: 11 },
  margin: { l: 60, r: 58, t: 12, b: 40 },
  hovermode: "x",
  showlegend: false,
  annotations: emphasisAnnotations(initial.maxDay),
  xaxis: {
    type: "log", range: initial.x,
    tickvals: ticks.vals, ticktext: ticks.text,
    gridcolor: "rgba(255,255,255,.06)", zeroline: false,
    showspikes: true, spikemode: "across", spikethickness: 1,
    spikecolor: "rgba(245,179,1,.8)", spikedash: "solid", spikesnap: "cursor",
  },
  yaxis: {
    type: "log", range: initial.y,
    tickformat: "$,.3~s",
    gridcolor: "rgba(255,255,255,.06)", zeroline: false,
  },
};

const config = { displayModeBar: false, responsive: true, scrollZoom: false };

document.getElementById("meta").textContent =
  "Fitted " + NQ + " quantile regressions on daily closes through " + P.last_date +
  " · genesis " + P.genesis + " · built " + P.generated_at.slice(0, 10);

window.__chartReady = Plotly.newPlot(gd, buildTraces(), layout, config);
</script>
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/ -v`
Expected: PASS, 26 tests

- [ ] **Step 5: Commit**

```bash
git add chart_template.html tests/test_chart_html.py
git commit -m "feat: draw the 99-quantile band, emphasis lines, and price series"
```

---

### Task 8: The hover readout panel

**Files:**
- Modify: `chart_template.html`
- Modify: `tests/test_chart_html.py`

**Interfaces:**
- Consumes: `quantilePricesAt`, `qIndex`, `fmtPrice`, `fmtDate`, `P` from Task 7.
- Produces: browser globals `window.__updatePanel(days)` and `window.__panelState` (an object with `days`, `pinned`, `step`), relied on by Tasks 9 and 10.

- [ ] **Step 1: Write the failing structural test**

Append to `tests/test_chart_html.py`:

```python
def test_template_exposes_the_panel_hooks():
    text = TEMPLATE.read_text(encoding="utf-8")
    assert "window.__updatePanel" in text
    assert "window.__panelState" in text
    assert "plotly_hover" in text
    assert "plotly_click" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_chart_html.py -k panel_hooks -v`
Expected: FAIL — `assert "window.__updatePanel" in text`

- [ ] **Step 3: Add the panel script**

Append inside the same `<script>` block in `chart_template.html`, after the `Plotly.newPlot` line:

```javascript
const panelDate = document.getElementById("panel-date");
const panelSub = document.getElementById("panel-sub");
const panelActual = document.getElementById("panel-actual");
const panelRows = document.getElementById("panel-rows");

const state = { days: null, pinned: false, step: 5 };
window.__panelState = state;

// P.days is ascending; return its index for an exact day, or -1 when the day
// falls outside the recorded series (i.e. we are in the projection).
function dayIndex(day) {
  let lo = 0, hi = P.days.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (P.days[mid] === day) return mid;
    if (P.days[mid] < day) lo = mid + 1; else hi = mid - 1;
  }
  return -1;
}

function displayedRows() {
  const rows = [];
  for (let i = NQ - 1; i >= 0; i -= state.step) rows.push(i);
  return rows;
}

function updatePanel(rawDays) {
  const day = Math.max(1, Math.round(rawDays));
  state.days = day;

  const prices = quantilePricesAt(day);
  const idx = dayIndex(day);
  const hasActual = idx !== -1;

  panelDate.textContent = fmtDate(day);
  panelSub.textContent = hasActual
    ? "Day " + day + " since genesis"
    : "Projected · day " + day + " since genesis";

  if (hasActual) {
    const close = P.close[idx];
    panelActual.hidden = false;
    panelActual.innerHTML =
      '<div class="price">' + fmtPrice(close) + "</div>" +
      '<div class="q">actual close · sits at q' + P.closest_q[idx].toFixed(2) + "</div>";
  } else {
    panelActual.hidden = true;
  }

  const actual = hasActual ? P.close[idx] : null;
  const rows = displayedRows();

  // Highlight the displayed pair that brackets the actual price.
  let upperBracket = -1;
  if (actual !== null) {
    for (let k = 0; k < rows.length; k++) {
      if (prices[rows[k]] >= actual) upperBracket = k;
    }
  }

  panelRows.innerHTML = rows
    .map(function (i, k) {
      const bracket =
        upperBracket !== -1 && (k === upperBracket || k === upperBracket + 1);
      return (
        '<tr class="' + (bracket ? "bracket" : "") + '">' +
        '<td class="q">q' + P.quantiles[i].toFixed(2) + "</td>" +
        '<td class="p">' + fmtPrice(prices[i]) + "</td>" +
        "</tr>"
      );
    })
    .join("");
}
window.__updatePanel = updatePanel;

gd.on("plotly_hover", function (ev) {
  if (state.pinned) return;
  const pt = ev.points[ev.points.length - 1];
  if (pt) updatePanel(pt.x);
});

gd.on("plotly_click", function (ev) {
  state.pinned = !state.pinned;
  panelSub.style.color = state.pinned ? "var(--accent)" : "";
  const pt = ev.points[ev.points.length - 1];
  if (state.pinned && pt) updatePanel(pt.x);
});

window.__chartReady.then(function () { updatePanel(LAST_DAY); });
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/ -v`
Expected: PASS, 27 tests

- [ ] **Step 5: Commit**

```bash
git add chart_template.html tests/test_chart_html.py
git commit -m "feat: add hover readout panel with per-quantile prices"
```

---

### Task 9: Horizon, density, and pin controls

**Files:**
- Modify: `chart_template.html`
- Modify: `tests/test_chart_html.py`

**Interfaces:**
- Consumes: `axisRanges`, `emphasisAnnotations`, `updatePanel`, `state` from Tasks 7-8.
- Produces: browser global `window.__setHorizon(years)`; the density buttons mutate `window.__panelState.step`.

- [ ] **Step 1: Write the failing structural test**

Append to `tests/test_chart_html.py`:

```python
def test_template_exposes_the_horizon_control():
    text = TEMPLATE.read_text(encoding="utf-8")
    assert "window.__setHorizon" in text
    assert "Plotly.relayout" in text


def test_template_offers_four_horizons_and_three_densities():
    text = TEMPLATE.read_text(encoding="utf-8")
    for years in ("0", "2", "5", "10"):
        assert f'data-years="{years}"' in text
    for step in ("1", "5", "10"):
        assert f'data-step="{step}"' in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_chart_html.py -k "horizon_control or four_horizons" -v`
Expected: FAIL — `assert "window.__setHorizon" in text`

- [ ] **Step 3: Add the control script**

Append inside the same `<script>` block in `chart_template.html`:

```javascript
function selectInGroup(group, button) {
  group.querySelectorAll("button").forEach(function (b) {
    b.setAttribute("aria-pressed", String(b === button));
  });
}

// The band is always built out to MAX_DAY, so changing horizon is a pure
// relayout: no refit, no trace rebuild.
function setHorizon(years) {
  const r = axisRanges(years);
  return Plotly.relayout(gd, {
    "xaxis.range": r.x,
    "yaxis.range": r.y,
    annotations: emphasisAnnotations(r.maxDay),
  });
}
window.__setHorizon = setHorizon;

const horizonGroup = document.getElementById("horizon");
horizonGroup.addEventListener("click", function (ev) {
  const button = ev.target.closest("button");
  if (!button) return;
  selectInGroup(horizonGroup, button);
  setHorizon(Number(button.dataset.years));
});

const densityGroup = document.getElementById("density");
densityGroup.addEventListener("click", function (ev) {
  const button = ev.target.closest("button");
  if (!button) return;
  selectInGroup(densityGroup, button);
  state.step = Number(button.dataset.step);
  updatePanel(state.days === null ? LAST_DAY : state.days);
});
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/ -v`
Expected: PASS, 29 tests

- [ ] **Step 5: Commit**

```bash
git add chart_template.html tests/test_chart_html.py
git commit -m "feat: add horizon, readout density, and pin controls"
```

---

### Task 10: Wire up the CLI, build with real data, verify in a browser

**Files:**
- Modify: `build_chart.py`
- Modify: `README.md`
- Modify: `CLAUDE.md`
- Create: `docs/index.html` (generated)

**Interfaces:**
- Consumes: everything above.
- Produces: `main(argv=None) -> int` and a committed `docs/index.html`.

- [ ] **Step 1: Add the CLI**

Append to `build_chart.py` (import `import argparse` and `import sys` at the top):

```python
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
```

- [ ] **Step 2: Verify the CLI end-to-end on a synthetic CSV**

Write a throwaway CSV so the pipeline runs without touching Kaggle:

```bash
python - <<'PY'
import numpy as np, pandas as pd
days = pd.date_range("2012-01-01", "2026-08-01", freq="D")
d = (days - pd.Timestamp("2010-01-03")).days.to_numpy()
rng = np.random.default_rng(7)
close = np.exp(-9.5 + 1.9*np.log(d) + rng.normal(0, 0.5, len(d)).cumsum()*0.02)
pd.DataFrame({
    "Timestamp": days.astype("int64")//10**9,
    "Open": close, "High": close*1.02, "Low": close*0.98,
    "Close": close, "Volume": 1.0,
}).to_csv("/tmp/synthetic_minutes.csv", index=False)
print("wrote /tmp/synthetic_minutes.csv")
PY

python build_chart.py --csv /tmp/synthetic_minutes.csv --out /tmp/synthetic_index.html
```

Expected: prints row count, pseudo R-squared, and `Wrote /tmp/synthetic_index.html`. Any crossing warning is informational, not a failure.

- [ ] **Step 3: Build from real Kaggle data**

Run: `python build_chart.py`
Expected: `docs/index.html` written, roughly 250-400 KB, with a printed latest close and its quantile.

- [ ] **Step 4: Verify in a browser**

Open `docs/index.html` in Chrome (via the claude-in-chrome tools or by hand) and confirm each of these:

1. The gradient band renders from cool at the bottom to warm at the top, with 11 labeled emphasis lines and the price line on top.
2. Hovering anywhere moves the crosshair and repaints the panel.
3. Hovering **past the last price** still shows projected per-quantile prices, and hides the "actual close" block.
4. Clicking pins the panel; clicking again releases it.
5. The horizon buttons change the visible range without a visible redraw of the band.
6. The density buttons switch between 11, 21, and 99 rows.

Then run the parity check in the browser console — this is the spec's success criterion, that the browser's numbers match Python's:

```javascript
await window.__chartReady;
JSON.stringify(
  [0.05, 0.25, 0.50, 0.75, 0.95].map(q => {
    const i = PAYLOAD.quantiles.indexOf(q);
    return [q, window.__quantilePricesAt(6000)[i]];
  })
);
```

And compare against Python:

```bash
python - <<'PY'
import numpy as np, json, pathlib, re
html = pathlib.Path("docs/index.html").read_text()
payload = json.loads(re.search(r"const PAYLOAD = (\{.*?\});", html, re.S).group(1))
coef = np.array(payload["coef"])
prices = np.sort(np.exp(coef[:, 0] + coef[:, 1] * np.log(6000)))
for q in (0.05, 0.25, 0.50, 0.75, 0.95):
    i = payload["quantiles"].index(q)
    print(f"q{q:.2f} {prices[i]:,.2f}")
PY
```

Expected: the two sets of numbers agree to at least 6 significant figures. If they do not, the JS and Python evaluation have diverged — stop and reconcile before continuing.

- [ ] **Step 5: Document it**

Append to `README.md`:

````markdown
## Interactive quantile chart

`docs/index.html` is a self-contained interactive chart of all 99 fitted quantile
bands. Hover any date — historical or projected — to read the price at every
quantile, and see where the actual close sat in the distribution.

Rebuild it with fresh Kaggle data:

```bash
python build_chart.py
```

Options: `--csv PATH` to skip the Kaggle download, `--out PATH` to write elsewhere.
The output has no local dependencies; open it directly from disk.
````

Add to `CLAUDE.md` under **Key Components**:

```markdown
- **build_chart.py**: Fits the 99 quantile regressions and bakes them into a
  self-contained interactive HTML chart at `docs/index.html`. Run `python build_chart.py`
  to refresh. Design: `docs/superpowers/specs/2026-08-30-interactive-quantile-chart-design.md`
- **chart_template.html**: Browser UI for the chart; `/*__PAYLOAD__*/` is the data marker.
```

- [ ] **Step 6: Run the full suite**

Run: `python -m pytest tests/ -v`
Expected: PASS, 29 tests

- [ ] **Step 7: Commit**

```bash
git add build_chart.py docs/index.html README.md CLAUDE.md
git commit -m "feat: build interactive quantile chart from live Kaggle data"
```

---

## Self-Review Notes

Checked against the spec:

- **Architecture / build_chart.py functions** — Tasks 1-6 and 10 cover `download_minute_csv`, `to_daily`, `fit_quantiles`, `check_crossings`, `closest_quantile`, `build_payload`, `render`, `main`. The spec listed `load_daily`; it is split into `download_minute_csv` (I/O) plus `to_daily` (pure) so the aggregation is testable without network. Same behaviour, better seam.
- **Payload** — Task 5, all eleven keys including `crossings_in_range`.
- **Chart: axes, band, emphasis lines, price** — Task 7.
- **Hover: event source, panel contents, crosshair, touch** — Task 8, plus the responsive breakpoint in the Task 6 CSS.
- **Controls: horizon, density, pin** — Task 9.
- **Quantile crossing** — detection in Task 3, browser-side sorting in `quantilePricesAt` (Task 7), conditional band sampling via `BAND_X` (Task 7), warning output in Task 10.
- **Genesis date** — Task 1, one constant with the discrepancy noted in a comment; asserted by `test_genesis_date_matches_notebook`.
- **Error handling** — Kaggle failure (Task 1), `QuantReg` non-convergence (Task 2), non-positive prices (Task 1), missing marker (Task 6).
- **Testing** — all five spec bullets are covered; the spec's "monotonicity after `fix_crossings`" became `check_crossings` detection tests plus browser-side sort, following the spec's own revision.
- **Success criteria 1-5** — Task 10 steps 3, 4, and the parity check.

Naming is consistent across tasks: `quantilePricesAt`, `qIndex`, `updatePanel`, `axisRanges`, `emphasisAnnotations`, `state`/`window.__panelState`, `BAND_X`/`BAND_Y`, `MAX_DAY`, `LAST_DAY`, `FIRST_DAY` are each defined once and referenced with the same spelling everywhere.
