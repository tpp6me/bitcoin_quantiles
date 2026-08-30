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
