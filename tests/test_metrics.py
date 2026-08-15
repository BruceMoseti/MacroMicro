"""Performance statistics, against hand-calculable cases."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, metrics


def _series(values, start="2020-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="B"), dtype=float)


def test_annualisation_of_a_constant_return():
    returns = _series([0.001] * config.TRADING_DAYS_PER_YEAR)
    stats = metrics.performance(returns)
    assert stats["annual_return"] == pytest.approx(0.001 * config.TRADING_DAYS_PER_YEAR)
    assert stats["annual_vol"] == pytest.approx(0.0, abs=1e-12)
    assert stats["years"] == pytest.approx(1.0)


def test_sharpe_matches_the_definition():
    rng = np.random.default_rng(3)
    returns = _series(rng.normal(0.0004, 0.01, 1000))
    stats = metrics.performance(returns)
    expected = returns.mean() * 252 / (returns.std() * np.sqrt(252))
    assert stats["sharpe"] == pytest.approx(expected)


def test_max_drawdown_on_a_known_path():
    # Cumulative log path: 0.10, 0.05, 0.15. Peak 0.10, trough 0.05 -> drawdown 0.05.
    returns = _series([0.10, -0.05, 0.10])
    assert metrics.max_drawdown(returns) == pytest.approx(-0.05)


def test_hit_rate_ignores_flat_days():
    """A strategy right on every day it holds risk has a 100% hit rate, however often it is flat.

    Risk is held on days where the *lagged* position is non-zero, because that is the
    position the return accrues to.
    """
    positions = _series([1.0, 1.0, 1.0, 0.0, 1.0, 0.0])  # lagged: 0, 1, 1, 1, 0, 1
    returns = _series([0.0, 0.01, 0.02, 0.03, 0.0, 0.04])
    stats = metrics.performance(returns, positions=positions)
    assert stats["active_days"] == 4
    assert stats["hit_rate"] == pytest.approx(1.0)
    # Measured over all days instead, the same strategy would look 67% right at best.
    assert (returns > 0).mean() == pytest.approx(4 / 6)


def test_profit_factor_and_win_loss():
    returns = _series([0.02, -0.01, 0.04, -0.01])
    stats = metrics.performance(returns)
    assert stats["profit_factor"] == pytest.approx(0.06 / 0.02)
    assert stats["avg_win"] == pytest.approx(0.03)
    assert stats["avg_loss"] == pytest.approx(-0.01)


def test_excess_returns_subtract_the_daily_cash_rate():
    returns = _series([0.001] * 10)
    rate = _series([2.52] * 10)  # 2.52% annual -> 0.01% per day at 252 days
    excess = metrics.excess_returns(returns, rate)
    assert excess.iloc[0] == pytest.approx(0.001 - 0.0252 / 252)


def test_turnover_and_exposure():
    returns = _series([0.0] * 5)
    positions = _series([0.0, 1.0, 1.0, -1.0, 0.0])
    stats = metrics.performance(returns, positions=positions)
    assert stats["market_exposure"] == pytest.approx(3 / 5)
    assert stats["avg_abs_position"] == pytest.approx(0.6)


def test_beta_to_benchmark_recovers_a_known_beta():
    rng = np.random.default_rng(11)
    benchmark = _series(rng.normal(0, 0.01, 800))
    strategy = 0.5 * benchmark + _series(rng.normal(0, 0.001, 800))
    stats = metrics.performance(strategy, benchmark=benchmark)
    assert stats["beta_to_benchmark"] == pytest.approx(0.5, abs=0.03)


def test_rolling_relationships_recovers_a_known_correlation():
    rng = np.random.default_rng(5)
    a = _series(rng.normal(0, 1, 600))
    frame = pd.DataFrame({"a": a, "b": a})
    rolling = metrics.rolling_relationships(frame, {"same": ("a", "b")})
    assert rolling["corr63_same"].dropna().round(6).eq(1.0).all()
    assert rolling["beta252_same"].dropna().round(6).eq(1.0).all()


def test_empty_input_reports_zero_observations():
    assert metrics.performance(pd.Series(dtype=float))["nobs"] == 0
