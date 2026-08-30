# Interactive Power-Law Quantile Chart — Design

Date: 2026-08-30
Status: Approved, not yet implemented

## Problem

The notebook (`bitcoin_quantile_studies_daily_kaggle_data.ipynb`) fits 99 quantile
regressions of Bitcoin's log price against log days since genesis, but renders them
as a static matplotlib PNG with a 9-line legend. Reading a value off it means
squinting at an axis.

We want an interactive log-log chart where hovering any date — historical or
projected — displays the price at every quantile for that date, and shows where the
actual price sat in the distribution. `charts.bitbo.io/long-term-power-law` is the
reference for feel, but it exposes only a handful of bands and no per-quantile
readout. This shows far more.

## Key insight

The fitted model is 99 pairs of numbers. Cell 38 of the notebook fits

    log_Close ~ log_days_since_genesis

at q = 0.01 … 0.99 and keeps an intercept and slope for each. Every band is

    price(date, q) = exp( a_q + b_q * ln(days_since_genesis) )

So the browser needs 198 floats, not a model. Three consequences:

1. The hover readout is **exact**, not interpolated between drawn lines.
2. Projection into the future is free — evaluate the same closed form at a larger
   `days`. No refit, no extrapolation machinery.
3. In log-log space each quantile line is **straight**, so a band is two points
   rather than one per day (see "Quantile crossing" for the sampled case). The
   gradient renders cheaply either way.

## Non-goals

- No refit control in the browser. 99 quantiles is the ceiling; 0.005 steps would be
  fitting noise at this sample size.
- No halving-cycle or Random-Forest content. This is the power-law quantile view only;
  the ML section of the notebook is out of scope.
- Not a forecast. The projected region is the fitted power law extended forward,
  labeled as such, with no confidence claim attached.

## Decisions taken

| Question | Decision |
|---|---|
| Delivery | Standalone self-contained HTML committed to the repo |
| Quantile rendering | All 99 as a shaded gradient, ~11 emphasized labeled lines |
| Hover readout | Fixed side panel, every 5th quantile by default |
| Projection | Yes, with a horizon control (today / +2y / +5y / +10y) |
| Data refresh | `build_chart.py` rebuild; data baked into the HTML |

## Architecture

Two source files, one generated artifact.

```
bitcoin_quantiles/
  build_chart.py          # new — fetch, fit, render
  chart_template.html     # new — the entire UI, one PAYLOAD marker
  docs/index.html         # generated — self-contained, committed
  tests/test_build_chart.py
```

`docs/` rather than the repo root so GitHub Pages can serve the chart later without
restructuring. (The spec directory lives under `docs/` too; Pages serving it is
harmless.)

### build_chart.py

Pure functions, each independently testable, plus a thin `main()`.

- `load_daily() -> DataFrame`
  kagglehub download of `mczielinski/bitcoin-historical-data`, read
  `btcusd_1-min_data.csv`, group by date, aggregate OHLCV. Lifted from notebook
  cells 3–8. Adds `DaysSinceGenesis`, `log_Close`, `log_days_since_genesis`.

- `fit_quantiles(daily) -> list[tuple[float, float]]`
  The cell 26/38 loop. `sm.QuantReg(Y, sm.add_constant(X)).fit(q=q)` for
  `q in np.arange(0.01, 1.00, 0.01)`. Returns 99 `(intercept, slope)` pairs in
  quantile order.

- `check_crossings(coefs, quantiles, day_range) -> list[Crossing]`
  Detection only; see "Quantile crossing" below.

- `closest_quantile(daily, coefs) -> ndarray`
  Cell 40's argmin over predicted `log_Close`, precomputed here so the browser does
  not repeat it per hover.

- `build_payload(daily, coefs, closest) -> dict`

- `render(payload, template_path, out_path)`
  Read template, replace the `/*__PAYLOAD__*/` marker with `json.dumps(payload)`,
  write output. Plain string replacement — no Jinja dependency.

- `main()` — wire them together, print a summary (row count, date range, fit
  R-squared at the median, any crossing warnings).

### Payload

Roughly 250 KB with prices rounded to 2dp and days stored as ints.

```json
{
  "generated_at": "2026-08-30T00:00:00Z",
  "genesis": "2010-01-03",
  "last_date": "2026-08-29",
  "last_close": 0.0,
  "quantiles": [0.01, 0.02, "..."],
  "coef": [[intercept, slope], "..."],
  "days":  [365, 366, "..."],
  "close": [0.06, 0.07, "..."],
  "closest_q": [0.42, "..."],
  "crossings_in_range": false
}
```

`days` and `close` are parallel arrays over the daily series. Dates are derived in
JS from `genesis + days`, so no date strings are stored per row.

## The chart

**Axes.** Log price on y against log days-since-genesis on x, with x tick *labels*
rewritten as calendar years — it reads as a date axis while staying a true log-log
plot, the same trick as notebook cell 20.

**The band.** 98 `fill: 'tonexty'` traces between consecutive quantile lines,
colored on a continuous scale running cool at q0.01 to warm at q0.99. Two points per
trace.

**Emphasis lines.** Overdrawn at q = 0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75,
0.90, 0.95, 0.99, each labeled at the right edge.

**Price.** The actual daily close as a thin dark line on top of the band.

## Hover

A tooltip cannot hold 21 rows, so the readout is a fixed side panel rather than a
bubble.

**Event source.** One invisible dense-x-grid trace plus `hovermode: 'x'` makes
Plotly fire `plotly_hover` anywhere on the canvas, including the empty projection
region to the right of the last price point. This is the supported API path; if it
proves unreliable, the fallback is a `mousemove` listener converting pixels to data
coordinates via the x-axis `p2d`.

**On each event.** Read x, convert to days and then to a date, evaluate all 99
quantiles in JS, repaint the panel.

**Panel contents.**
- The hovered date.
- Actual close and its quantile — historical dates only; omitted in the projection.
- Price at every 5th quantile, q0.95 down to q0.05, with the row pair bracketing the
  actual price highlighted.

**Crosshair.** `xaxis.showspikes` — a Plotly built-in, so no `relayout` call per
mousemove.

**Touch.** The same handler fires on tap; the panel sits below the chart rather than
beside it under a narrow viewport.

## Controls

- **Horizon** — today / +2y / +5y / +10y. An `xaxis.range` relayout; nothing refits.
- **Readout density** — every 5th quantile (21 rows) / every 10th (11 rows) / all 99
  in a scrolling panel.
- **Pin** — click to freeze the panel so it can be read without holding the mouse
  still. Click again to release.

## Quantile crossing

The 99 lines are fitted independently, so nothing constrains q0.90 to sit above q0.75
at every date. Independently fitted quantile lines with differing slopes must cross
somewhere; the question is only whether the crossing falls inside the displayed
range, and the +10y horizon widens that range well past the data.

Rearrangement happens **at evaluation time, in the browser**: after evaluating all 99
quantiles at a given x, sort the resulting prices ascending before displaying them.
That is exactly the standard rearrangement fix, costs one 99-element sort per hover,
and preserves the closed-form payload. The emitted coefficients are never modified.

The build script's job is therefore detection, not repair. `check_crossings(coefs,
quantiles, day_range)` evaluates all 99 lines across the full +10y day range and
returns the list of inverted pairs with the day at which each inversion begins. It
prints a warning naming them and sets `"crossings_in_range": true` in the payload.

The drawn band respects the same flag:

- **No crossings in range** — each band is a 2-point trace, since the lines are
  straight in log-log space.
- **Crossings in range** — bands are sampled at 200 x positions across the range and
  sorted at each, giving 98 polylines of 200 points. Still well inside what Plotly
  renders comfortably, and it keeps the drawn band consistent with the panel numbers.

Without rearrangement the panel could show a higher quantile at a lower price, which
reads as a bug rather than as a property of the fit.

## Genesis date

The notebook uses **2010-01-03** for `DaysSinceGenesis` (cell 14), but 2009-01-09 in
its halving list (cell 44), and `bitcoin_quantile_fixes.py` uses 2009-01-03. The
chart keeps **2010-01-03** so its numbers match the existing analysis, held in one
named constant, `GENESIS_DATE`, with a comment recording the discrepancy.

## Error handling

- Kaggle download failure — fail loudly with the kagglehub error and a note that the
  cached copy under `~/.cache/kagglehub` is used automatically when present.
- `QuantReg` non-convergence at an extreme quantile — catch per quantile, report which
  q failed, abort rather than emit a partial band.
- Zero or negative prices in the source data — filter before the log transform and
  report how many rows were dropped.
- Template marker missing — fail with the expected marker string.

## Testing

`tests/test_build_chart.py`, pytest, added to `requirements.txt`.

- `fit_quantiles` recovers known coefficients on synthetic `y = a + b*ln(x)` data.
- Payload schema: 99 quantiles, 99 coefficient pairs, `days` and `close` and
  `closest_q` of equal length, no NaN anywhere.
- `check_crossings` finds the inversion in a hand-built pair of deliberately crossing
  lines, and finds none in a non-crossing pair.
- Sorted evaluation is monotone at every sampled x across the full +10y horizon.
- `closest_quantile` on a hand-built frame puts a point known to sit on the q0.50 line
  at q0.50.
- Generated HTML is self-contained: contains the payload, and references no local
  file paths that would break when opened over `file://`.

Then a browser pass with claude-in-chrome: hover a known date, confirm the panel's
numbers match `build_chart.py`'s own prediction for that date, and check the
projection region, the horizon control, and pinning.

## Success criteria

1. `python build_chart.py` regenerates `docs/index.html` from current Kaggle data.
2. Opening that file directly from disk shows the chart with no server and no network.
3. Hovering any date shows per-quantile prices, and where the actual price sat.
4. Hovering past the last price shows projected per-quantile prices.
5. The quantile ordering in the panel is monotone at every date in range.
