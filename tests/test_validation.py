"""Validation checks must fail on broken data.

Each test corrupts one property of a good dataset and asserts that the corresponding
check catches it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, validation


def _log():
    return validation.ValidationLog()


def test_clean_panel_passes(panel):
    log = _log()
    validation.check_daily_panel(log, panel)
    assert not log.failures, [c.detail for c in log.failures]


def test_duplicate_timestamps_are_caught(panel):
    corrupted = pd.concat([panel, panel.iloc[[-1]]])
    log = _log()
    validation.check_daily_panel(log, corrupted)
    assert any(c.name == "daily.no_duplicate_timestamps" for c in log.failures)


def test_future_dates_are_caught(panel):
    extra = panel.iloc[[-1]].copy()
    extra.index = [config.END_DATE + pd.Timedelta(days=7)]
    log = _log()
    validation.check_daily_panel(log, pd.concat([panel, extra]))
    assert any(c.name == "daily.no_future_dates" for c in log.failures)


def test_non_positive_prices_are_caught(panel):
    """The 2020-04-20 negative WTI print is the real-world version of this."""
    corrupted = panel.copy()
    corrupted.loc[corrupted.index[500], "wti"] = -37.63
    log = _log()
    validation.check_daily_panel(log, corrupted)
    failed = [c for c in log.failures if c.name == "daily.positive_prices"]
    assert failed and "wti" in failed[0].detail


def test_unsorted_index_is_caught(panel):
    log = _log()
    validation.check_daily_panel(log, panel.iloc[::-1])
    assert any(c.name == "daily.monotonic_index" for c in log.failures)


def test_anchor_mismatch_is_caught(panel):
    corrupted = panel.copy()
    corrupted.loc[config.ANCHOR_DATE, "dgs10"] = 9.99
    log = _log()
    validation.check_anchor_values(log, corrupted, config.SOURCE_SYNTHETIC)
    names = {c.name for c in log.failures}
    assert "anchors.dgs10" in names
    assert "anchors.curve_2s10s" in names


def test_anchor_values_pass_on_the_real_panel(panel):
    log = _log()
    validation.check_anchor_values(log, panel, config.SOURCE_SYNTHETIC)
    assert not log.failures
    detail = next(c.detail for c in log.checks if c.name == "anchors.curve_2s10s")
    assert "48" in detail


def test_broken_open_interest_identity_is_caught(dataset):
    corrupted = dataset.cot.copy()
    corrupted.loc[corrupted.index[0], "long"] = corrupted.loc[corrupted.index[0], "long"] * 3
    log = _log()
    validation.check_cot(log, corrupted)
    assert any(c.name == "cot.reconciles_to_open_interest" for c in log.failures)


def test_non_tuesday_cot_report_is_caught(dataset):
    corrupted = dataset.cot.copy()
    corrupted.loc[corrupted.index[0], "report_date"] = pd.Timestamp("2020-01-01")  # a Wednesday
    log = _log()
    validation.check_cot(log, corrupted)
    assert any(c.name == "cot.as_of_is_tuesday" for c in log.failures)


def test_negative_positions_are_caught(dataset):
    corrupted = dataset.cot.copy()
    corrupted.loc[corrupted.index[0], "short"] = -5.0
    log = _log()
    validation.check_cot(log, corrupted)
    assert any(c.name == "cot.no_negative_positions" for c in log.failures)


def test_vintage_before_reference_period_is_caught(dataset):
    corrupted = dataset.macro_vintages.copy()
    corrupted.loc[corrupted.index[0], "vintage_date"] = pd.Timestamp("1990-01-01")
    log = _log()
    validation.check_macro_vintages(log, corrupted)
    assert any(c.name == "macro.vintage_after_reference_period" for c in log.failures)


def test_infinite_returns_are_caught():
    returns = pd.Series([0.01, np.inf, -0.02])
    log = _log()
    validation.check_returns_finite(log, "broken", returns)
    assert log.failures


def test_regression_without_sample_size_is_caught():
    log = _log()
    validation.check_regression_reports_n(log, pd.DataFrame({"term": ["a"], "coefficient": [1.0]}))
    assert log.failures


def test_backtest_without_trade_count_is_caught():
    log = _log()
    validation.check_backtest_reports_trades(log, pd.DataFrame({"signal": ["a"], "sharpe": [1.0]}))
    assert log.failures


def test_undated_chart_is_caught():
    log = _log()
    validation.check_charts_dated(log, pd.DataFrame({"chart": ["x.png"], "dated": [False]}))
    assert log.failures


def test_assert_ok_raises_only_on_critical_failures():
    log = _log()
    log.add("advisory_only", False, "a warning", critical=False)
    log.assert_ok()  # advisory failures must not abort the pipeline
    log.add("critical_failure", False, "a real problem", critical=True)
    with pytest.raises(validation.ValidationError):
        log.assert_ok()


def test_summary_counts_results():
    log = _log()
    log.add("a", True, "ok")
    log.add("b", False, "warn", critical=False)
    log.add("c", False, "fail")
    frame = log.frame()
    assert set(frame["result"]) == {"PASS", "WARN", "FAIL"}
