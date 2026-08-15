"""Leakage tests.

Two things are checked here. First, that the platform's own transforms are causal.
Second, and more importantly, that the causality check itself has teeth: deliberately
leaky transforms are fed to it and must fail. A leakage test that cannot fail is worse
than no leakage test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import backtest, features, relative_value, signals, validation


def _log() -> validation.ValidationLog:
    return validation.ValidationLog()


# --------------------------------------------------------------------------------------
# The check must catch leakage
# --------------------------------------------------------------------------------------
def test_causality_check_rejects_a_full_sample_zscore(panel):
    def leaky(frame: pd.DataFrame) -> pd.DataFrame:
        # Full-sample mean and sd: every row depends on every other row.
        return (frame - frame.mean()) / frame.std()

    log = _log()
    validation.check_causality(log, "full_sample_zscore", leaky, panel)
    assert log.failures, "a full-sample z-score must fail the causality check"


def test_causality_check_rejects_a_centred_window(panel):
    def leaky(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.rolling(21, center=True).mean()

    log = _log()
    validation.check_causality(log, "centred_window", leaky, panel)
    assert log.failures, "a centred rolling window must fail the causality check"


def test_causality_check_rejects_a_negative_shift(panel):
    def leaky(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.shift(-1)

    log = _log()
    validation.check_causality(log, "forward_shift", leaky, panel)
    assert log.failures, "reading tomorrow's value must fail the causality check"


def test_causality_check_accepts_a_trailing_window(panel):
    log = _log()
    validation.check_causality(log, "trailing_window", lambda f: f.rolling(21).mean(), panel)
    assert not log.failures


# --------------------------------------------------------------------------------------
# The platform's own transforms
# --------------------------------------------------------------------------------------
def test_market_features_are_causal(panel):
    log = _log()
    validation.check_causality(log, "market_features", features.build_feature_frame, panel)
    assert not log.failures, [c.detail for c in log.failures]


def test_cot_features_are_causal(dataset):
    log = _log()
    validation.check_causality(
        log, "cot_features", lambda p: features.cot_features(dataset.cot, p.index), dataset.daily
    )
    assert not log.failures, [c.detail for c in log.failures]


def test_relative_value_signal_is_causal(panel):
    def build(frame: pd.DataFrame) -> pd.DataFrame:
        result = relative_value.run(frame, features.curves(frame))
        return signals.relative_value_positions(result["zscores"])

    log = _log()
    validation.check_causality(log, "rv_signal", build, panel)
    assert not log.failures, [c.detail for c in log.failures]


def test_rolling_hedge_ratio_ignores_future_data(panel):
    pair = next(p for p in relative_value.PAIRS if p.name == relative_value.PRIMARY_PAIR)
    y, x = relative_value.pair_series(panel, features.curves(panel), pair)
    baseline = relative_value.rolling_residual(y, x)["beta"]
    shocked_y = y.copy()
    shocked_y.iloc[-100:] *= 1.5
    shocked = relative_value.rolling_residual(shocked_y, x)["beta"]
    pd.testing.assert_series_equal(baseline.iloc[:-100], shocked.iloc[:-100])


# --------------------------------------------------------------------------------------
# Backtest timing
# --------------------------------------------------------------------------------------
def test_position_cannot_earn_its_own_day_return():
    """A signal with perfect same-day knowledge must still make nothing."""
    index = pd.date_range("2020-01-01", periods=250, freq="B")
    returns = pd.Series(np.random.default_rng(7).normal(0, 0.01, len(index)), index=index)
    clairvoyant = np.sign(returns)  # today's position equals today's return sign
    result = backtest.run("clairvoyant_same_day", clairvoyant, returns)
    # Because positions are shifted, this is a one-day-lagged momentum rule, not an oracle.
    assert result.returns.sum() < returns.abs().sum() * 0.5
    expected = clairvoyant.shift(1).fillna(0.0) * returns
    pd.testing.assert_series_equal(result.gross_returns, expected, check_names=False)


def test_position_lagged_by_one_session():
    index = pd.date_range("2020-01-01", periods=6, freq="B")
    positions = pd.Series([0, 1, 1, -1, 0, 0], index=index, dtype=float)
    returns = pd.Series([0.01, 0.02, -0.03, 0.04, 0.05, -0.01], index=index)
    result = backtest.run("timing", positions, returns)
    # Day 1 has no prior position, so it earns nothing.
    assert result.gross_returns.iloc[0] == 0.0
    # Day 2 earns the position set on day 1, which was zero.
    assert result.gross_returns.iloc[1] == 0.0
    # Day 3 earns the position set on day 2.
    assert result.gross_returns.iloc[2] == 1.0 * -0.03
    assert result.gross_returns.iloc[3] == 1.0 * 0.04
    assert result.gross_returns.iloc[4] == -1.0 * 0.05


def test_true_oracle_would_be_detectable():
    """Sanity check on the previous test: a genuinely leaky backtest looks obviously wrong."""
    index = pd.date_range("2020-01-01", periods=250, freq="B")
    returns = pd.Series(np.random.default_rng(7).normal(0, 0.01, len(index)), index=index)
    leaked = (np.sign(returns) * returns).sum()  # no lag applied
    lagged = backtest.run("lagged", np.sign(returns), returns).returns.sum()
    assert leaked > 10 * abs(lagged)


def test_weights_are_finite_and_known(feature_frame):
    positioning = signals.positioning_positions(feature_frame)
    log = _log()
    for column in positioning.columns:
        validation.check_weights_known_in_advance(log, column, positioning[column])
    assert not log.failures
