"""Feature transforms and publication-aware alignment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, features, trading_calendar


def test_log_returns_match_hand_calculation():
    prices = pd.Series([100.0, 110.0, 99.0], index=pd.date_range("2020-01-01", periods=3))
    result = features.log_returns(prices)
    assert np.isnan(result.iloc[0])
    assert result.iloc[1] == pytest.approx(np.log(1.10))
    assert result.iloc[2] == pytest.approx(np.log(99.0 / 110.0))


def test_bp_changes_convert_percent_to_basis_points():
    yields = pd.Series([4.15, 4.63, 4.60], index=pd.date_range("2020-01-01", periods=3))
    result = features.bp_changes(yields)
    assert result.iloc[1] == pytest.approx(48.0)
    assert result.iloc[2] == pytest.approx(-3.0)


def test_curve_uses_the_anchor_convention(panel):
    curve = features.curves(panel)["curve_2s10s"]
    expected = (panel["dgs10"] - panel["dgs2"]) * config.BP_PER_PCT
    pd.testing.assert_series_equal(curve, expected, check_names=False)
    assert curve.loc[config.ANCHOR_DATE] == pytest.approx(config.ANCHOR_CURVE_2S10S_BP, abs=1.0)


def test_rolling_zscore_uses_only_trailing_data():
    values = pd.Series(np.arange(100.0), index=pd.date_range("2020-01-01", periods=100))
    result = features.rolling_zscore(values, window=20, min_periods=20)
    window = values.iloc[30:50]
    assert result.iloc[49] == pytest.approx((values.iloc[49] - window.mean()) / window.std())
    # A trailing window cannot produce a value before it is full.
    assert result.iloc[:19].isna().all()


def test_rolling_zscore_is_unaffected_by_later_observations():
    values = pd.Series(np.random.default_rng(0).normal(size=200), index=pd.date_range("2020-01-01", periods=200))
    baseline = features.rolling_zscore(values, 60)
    shocked = values.copy()
    shocked.iloc[150:] += 25.0
    assert baseline.iloc[:150].equals(features.rolling_zscore(shocked, 60).iloc[:150])


def test_cot_available_from_waits_for_the_session_after_friday_release():
    report_dates = pd.Series(pd.to_datetime(["2016-08-16", "2016-08-23", "2016-08-30"]))
    available = features.cot_available_from(report_dates)
    # Tuesday as-of, Friday 15:30 ET release, usable the next session: the following Monday.
    assert list(available.dt.weekday) == [0, 0, 1]
    lags = (available - report_dates).dt.days
    assert lags.tolist() == [6, 6, 7]  # 7 when that Monday is Labor Day


def test_cot_available_from_is_never_before_the_report_date(dataset):
    weekly = dataset.cot.drop_duplicates("report_date")[["report_date"]].reset_index(drop=True)
    available = features.cot_available_from(weekly["report_date"])
    assert (available.to_numpy() > weekly["report_date"].to_numpy()).all()
    sessions = set(trading_calendar.trading_days(config.START_DATE, config.END_DATE + pd.Timedelta(days=10)))
    assert set(available).issubset(sessions)


def test_cot_features_are_flat_between_publications(feature_frame):
    """A weekly series must not interpolate; it steps on publication and holds."""
    ratio = feature_frame["cot_zn_ratio"].dropna()
    changes = int((ratio.diff() != 0).sum())
    assert 400 < changes < 560, f"expected roughly one change per weekly report, got {changes}"


def test_macro_features_use_the_vintage_available_at_the_time(dataset):
    index = dataset.daily.index
    macro = features.macro_point_in_time(dataset.macro_vintages, index)
    assert not macro.empty
    first_vintage = dataset.macro_vintages["vintage_date"].min()
    before = macro.loc[macro.index < first_vintage]
    assert before.isna().all().all(), "no macro value may exist before the first vintage date"


def test_return_proxies_use_documented_duration(panel):
    proxies = features.return_proxies(panel)
    expected = -config.ZN_MODIFIED_DURATION * panel["dgs10"].diff() / config.BP_PER_PCT
    pd.testing.assert_series_equal(proxies["zn_proxy"], expected, check_names=False)
    pd.testing.assert_series_equal(
        proxies["usd_short_proxy"], -features.log_returns(panel["usd"]), check_names=False
    )


def test_feature_frame_has_no_infinities(feature_frame):
    numeric = feature_frame.select_dtypes("number").to_numpy(dtype=float)
    assert not np.isinf(numeric).any()
