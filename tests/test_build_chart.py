import numpy as np
import pandas as pd
import pytest

from build_chart import GENESIS_DATE, QUANTILES, to_daily, fit_quantiles


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


def test_to_daily_uses_chronological_open_and_close_when_rows_are_unordered():
    # Deliberately shuffle rows for a single day so chronologically-first row is not first in frame
    minute_df = _minute_frame(
        [
            ("2015-01-01 23:59", 20.0, 25.0, 19.0, 22.0, 2.0),
            ("2015-01-01 12:00", 15.0, 17.0, 14.0, 16.0, 1.5),
            ("2015-01-01 00:00", 10.0, 11.0, 9.0, 10.5, 1.0),
        ]
    )
    daily = to_daily(minute_df)

    assert len(daily) == 1
    # Open should be from the earliest time (00:00), not the first row (23:59)
    assert daily.loc[0, "Open"] == 10.0
    # Close should be from the latest time (23:59), not the last row (00:00)
    assert daily.loc[0, "Close"] == 22.0


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


from build_chart import Crossing, check_crossings, predict_log_prices, projection_days


def test_predict_log_prices_shape_and_values():
    coef = np.array([[0.0, 1.0], [1.0, 2.0]])
    days = np.array([10.0, 100.0, 1000.0])
    preds = predict_log_prices(coef, days)

    assert preds.shape == (3, 2)
    assert preds[0, 0] == pytest.approx(np.log(10.0))
    assert preds[1, 1] == pytest.approx(1.0 + 2.0 * np.log(100.0))
    assert preds[1, 0] == pytest.approx(np.log(100.0))


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
