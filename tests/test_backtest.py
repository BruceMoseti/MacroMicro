"""Backtester mechanics: costs, financing, sizing and trade counting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import backtest, config, signals


def _series(values, start="2020-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="B"), dtype=float)


def test_costs_charged_on_position_change_only():
    positions = _series([1.0, 1.0, 1.0, 0.0, 0.0])
    returns = _series([0.0] * 5)
    result = backtest.run("costs", positions, returns, cost_bps=10.0)
    # Cost lands on the session the trade is executed, one day after the signal changes.
    assert result.costs.iloc[1] == pytest.approx(10.0 / 10_000)
    assert result.costs.iloc[2] == 0.0  # holding is free
    assert result.costs.iloc[4] == pytest.approx(10.0 / 10_000)  # closing trade
    assert result.trades.sum() == 2


def test_zero_cost_equals_gross():
    positions = _series([0.0, 1.0, -1.0, 1.0, 0.0])
    returns = _series([0.01, -0.02, 0.03, -0.01, 0.02])
    result = backtest.run("free", positions, returns, cost_bps=0.0)
    pd.testing.assert_series_equal(result.returns, result.gross_returns, check_names=False)


def test_higher_costs_never_improve_returns():
    rng = np.random.default_rng(2)
    index = pd.date_range("2020-01-01", periods=600, freq="B")
    returns = pd.Series(rng.normal(0, 0.01, 600), index=index)
    positions = pd.Series(rng.choice([-1.0, 0.0, 1.0], 600), index=index)
    totals = [backtest.run("c", positions, returns, cost_bps=c).returns.sum() for c in config.COST_GRID_BPS]
    assert totals == sorted(totals, reverse=True)


def test_financing_is_not_charged_on_flat_days():
    positions = _series([0.0, 0.0, 0.0, 0.0])
    returns = _series([0.0, 0.0, 0.0, 0.0])
    rate = _series([5.0] * 4)
    result = backtest.run("flat", positions, returns, financing_pct=rate)
    assert result.returns.abs().sum() == pytest.approx(0.0)


def test_financing_reduces_a_long_and_helps_a_short():
    returns = _series([0.0, 0.0, 0.0])
    rate = _series([5.04] * 3)  # 0.02% per day
    long_result = backtest.run("long", _series([1.0, 1.0, 1.0]), returns, financing_pct=rate)
    short_result = backtest.run("short", _series([-1.0, -1.0, -1.0]), returns, financing_pct=rate)
    daily = 0.0504 / 252
    assert long_result.returns.iloc[-1] == pytest.approx(-daily)
    assert short_result.returns.iloc[-1] == pytest.approx(daily)


def test_vol_target_sizing_hits_the_target():
    rng = np.random.default_rng(4)
    index = pd.date_range("2018-01-01", periods=1500, freq="B")
    # Two regimes: 8% then 32% annualised volatility.
    daily = np.concatenate([rng.normal(0, 0.08 / np.sqrt(252), 750), rng.normal(0, 0.32 / np.sqrt(252), 750)])
    returns = pd.Series(daily, index=index)
    # Size is set at entry, so the signal has to open new trades to re-size. Toggle every
    # 20 sessions, starting after the volatility window is populated.
    states = pd.Series(0.0, index=index)
    states.iloc[100:] = np.tile(np.repeat([1.0, 0.0], 20), 100)[: len(index) - 100]

    sized = backtest.size_positions(states, returns, target_vol=0.10)
    held = sized.shift(1)
    active = held != 0
    realised = (held[active] * returns[active]).std() * np.sqrt(252)
    assert realised == pytest.approx(0.10, abs=0.035)
    # Trades opened in the high volatility regime must be sized smaller.
    assert sized.iloc[900:1400].max() < sized.iloc[100:600].max() / 2


def test_vol_target_leaves_a_trade_unsized_before_its_window_is_populated():
    """A trade opened before enough history exists is not sized, rather than guessing."""
    index = pd.date_range("2018-01-01", periods=400, freq="B")
    returns = pd.Series(0.01, index=index)
    states = pd.Series(1.0, index=index)  # open on day one and never change
    sized = backtest.size_positions(states, returns)
    assert (sized == 0).all()


def test_vol_target_sizing_is_held_for_the_life_of_the_trade():
    rng = np.random.default_rng(8)
    index = pd.date_range("2018-01-01", periods=400, freq="B")
    returns = pd.Series(rng.normal(0, 0.01, 400), index=index)
    states = pd.Series(0.0, index=index)
    states.iloc[200:240] = 1.0
    sized = backtest.size_positions(states, returns)
    held = sized.iloc[200:240]
    assert held.nunique() == 1, "size must be fixed at entry, not rescaled daily"
    assert (sized.iloc[:200] == 0).all()


def test_vol_target_respects_the_leverage_cap():
    index = pd.date_range("2018-01-01", periods=400, freq="B")
    returns = pd.Series(1e-9, index=index)  # near-zero volatility implies infinite scale
    states = pd.Series(1.0, index=index)
    sized = backtest.size_positions(states, returns)
    assert sized.max() <= backtest.MAX_LEVERAGE + 1e-9


def test_hysteresis_holds_between_the_bands():
    zscore = pd.Series([0.0, 2.5, 1.5, 0.9, 0.4, -1.0, -2.2, -0.2])
    positions = signals._hysteresis(zscore, entry=2.0, exit_level=0.5, direction=-1)
    # Enter short at 2.5, hold through 1.5 and 0.9, exit at 0.4, enter long at -2.2, exit at -0.2.
    assert positions.tolist() == [0.0, -1.0, -1.0, -1.0, 0.0, 0.0, 1.0, 0.0]


def test_hysteresis_flips_on_an_opposite_extreme():
    zscore = pd.Series([2.5, -2.5])
    positions = signals._hysteresis(zscore, entry=2.0, exit_level=0.5, direction=-1)
    assert positions.tolist() == [-1.0, 1.0]


def test_hysteresis_is_flat_while_the_zscore_is_missing():
    zscore = pd.Series([2.5, np.nan, 2.5])
    positions = signals._hysteresis(zscore, entry=2.0, exit_level=0.5, direction=-1)
    assert positions.tolist() == [-1.0, 0.0, -1.0]


def test_split_consistency_flags_sign_changes():
    headline = pd.DataFrame(
        {
            "signal": ["a"] * 3 + ["b"] * 3,
            "split": ["train", "validation", "test"] * 2,
            "sharpe": [0.5, 0.4, 0.3, -0.5, 0.4, 1.9],
        }
    )
    result = backtest.split_consistency(headline).set_index("signal")
    assert bool(result.loc["a", "positive_in_all_splits"])
    assert not bool(result.loc["b", "positive_in_all_splits"])
    assert int(result.loc["b", "splits_positive"]) == 2


def test_splits_partition_the_window_without_overlap():
    boundaries = [config.SPLITS[name] for name in ("train", "validation", "test")]
    for (_, earlier_end), (later_start, _) in zip(boundaries, boundaries[1:]):
        assert earlier_end < later_start
    assert boundaries[0][0] == config.START_DATE
    assert boundaries[-1][1] == config.END_DATE
