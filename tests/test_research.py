"""Research-layer properties: the frozen window, the collinearity trap, and the pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, features, regressions, relative_value, signals, synthetic, trading_calendar


# --------------------------------------------------------------------------------------
# The frozen dataset
# --------------------------------------------------------------------------------------
def test_research_window_is_frozen(panel):
    assert panel.index.min() == config.START_DATE
    assert panel.index.max() == config.END_DATE
    assert len(panel) == 2512, "the frozen window is 2,512 NYSE sessions"


def test_trading_calendar_excludes_known_holidays():
    days = set(trading_calendar.trading_days())
    for holiday in ("2025-01-01", "2025-01-20", "2025-04-18", "2025-06-19", "2025-11-27", "2025-12-25"):
        assert pd.Timestamp(holiday) not in days
    # Unscheduled closures.
    assert pd.Timestamp("2018-12-05") not in days
    assert pd.Timestamp("2025-01-09") not in days
    # A normal session.
    assert pd.Timestamp("2025-03-11") in days


def test_synthetic_dataset_is_deterministic():
    first = synthetic.build_daily_panel()
    second = synthetic.build_daily_panel()
    pd.testing.assert_frame_equal(first, second)


def test_synthetic_panel_hits_the_real_anchors(panel):
    for name, expected in config.ANCHORS.items():
        assert float(panel[name].iloc[-1]) == pytest.approx(expected, abs=config.ANCHOR_TOLERANCE[name])


def test_yield_identities_hold_exactly(panel):
    assert (panel["dgs10"] - (panel["real10"] + panel["breakeven10"])).abs().max() < 1e-9
    assert (panel[["dgs2", "dgs10", "dgs30"]] > 0).all().all(), "US nominal yields were never negative"


def test_synthetic_panel_is_economically_plausible(panel):
    """Guardrails on the simulator, so an implausible recalibration fails loudly."""
    returns = np.log(panel[["spx", "usd", "wti", "gold"]]).diff()
    annual_vol = returns.std() * np.sqrt(config.TRADING_DAYS_PER_YEAR)
    assert 0.10 < annual_vol["spx"] < 0.28
    assert 0.02 < annual_vol["usd"] < 0.10
    assert 0.20 < annual_vol["wti"] < 0.60
    assert 0.08 < annual_vol["gold"] < 0.25
    assert 8.0 < panel["vix"].median() < 25.0
    # Signs that must hold for the cross-asset story to make sense.
    assert returns["spx"].corr(np.log(panel["vix"]).diff()) < -0.4
    assert returns["gold"].corr(panel["real10"].diff()) < -0.2
    assert returns["usd"].corr(returns["wti"]) < 0.0


def test_curve_inverts_and_steepens(panel):
    curve = (panel["dgs10"] - panel["dgs2"]) * config.BP_PER_PCT
    assert (curve < 0).sum() > 100, "the window must contain a genuine inversion episode"
    assert curve.max() > 100


# --------------------------------------------------------------------------------------
# Hypothesis 1
# --------------------------------------------------------------------------------------
def test_naive_specification_is_rank_deficient(feature_frame):
    """dCurve = dy10 - dy2 exactly, so the textbook specification is not identified.

    The reported condition number must be exactly infinite rather than a large finite
    number: a finite value there is floating-point rounding error, and it does not
    reproduce across BLAS builds.
    """
    _, diagnostics = regressions.fit_ols(feature_frame, regressions.SPECIFICATIONS["naive"], label="naive")
    assert diagnostics["rank_deficient"]
    assert diagnostics["n_regressors"] == 4
    assert diagnostics["regressor_rank"] == 3
    assert diagnostics["condition_number"] == np.inf
    assert diagnostics["max_vif"] == np.inf
    assert diagnostics["fit_warnings"], "statsmodels should warn about the singular design"


def test_identified_specifications_are_well_conditioned(feature_frame):
    for name in ("slope", "real"):
        _, diagnostics = regressions.fit_ols(feature_frame, regressions.SPECIFICATIONS[name], label=name)
        assert not diagnostics["rank_deficient"]
        assert diagnostics["regressor_rank"] == diagnostics["n_regressors"]
        assert diagnostics["condition_number"] < 1e3
        assert diagnostics["max_vif"] < 10.0


def test_naive_and_slope_specifications_span_the_same_space(feature_frame):
    _, naive = regressions.fit_ols(feature_frame, regressions.SPECIFICATIONS["naive"], label="naive")
    _, slope = regressions.fit_ols(feature_frame, regressions.SPECIFICATIONS["slope"], label="slope")
    assert naive["r_squared"] == pytest.approx(slope["r_squared"], abs=1e-9)


def test_equity_signs_match_theory(feature_frame):
    coefficients, _ = regressions.fit_ols(feature_frame, regressions.SPECIFICATIONS["real"], label="real")
    terms = coefficients.set_index("term")
    assert terms.loc["dy_1d_real10", "coefficient"] < 0, "higher real discount rate hurts equities"
    assert terms.loc["dy_1d_breakeven10", "coefficient"] > 0, "reflation helps equities"
    assert terms.loc["ret_1d_vix", "coefficient"] < 0, "rising insurance cost coincides with selling"


def test_hac_bandwidth_follows_newey_west():
    assert regressions.hac_lags(2511) == 8
    assert regressions.hac_lags(100) == 4


def test_holm_bonferroni_is_monotone_and_conservative():
    p_values = pd.Series({"a": 0.001, "b": 0.02, "c": 0.30})
    result = regressions.holm_bonferroni(p_values)
    assert list(result["p_holm"]) == sorted(result["p_holm"])
    assert (result["p_holm"] >= result["p_value"]).all()
    assert result["p_holm"].iloc[0] == pytest.approx(0.003)


# --------------------------------------------------------------------------------------
# Hypothesis 2
# --------------------------------------------------------------------------------------
def test_half_life_recovers_a_known_ar1():
    rng = np.random.default_rng(1)
    phi = 0.9
    values = np.zeros(20_000)
    for i in range(1, len(values)):
        values[i] = phi * values[i - 1] + rng.normal(0, 0.1)
    estimated_phi, half_life = relative_value.half_life(pd.Series(values))
    assert estimated_phi == pytest.approx(phi, abs=0.02)
    assert half_life == pytest.approx(-np.log(2) / np.log(phi), rel=0.15)


def test_half_life_is_infinite_for_a_random_walk():
    rng = np.random.default_rng(2)
    walk = pd.Series(np.cumsum(rng.normal(size=5000)))
    _, half_life = relative_value.half_life(walk)
    assert half_life > 500 or not np.isfinite(half_life)


def test_spread_returns_equal_the_residual_change_for_a_constant_beta():
    index = pd.date_range("2020-01-01", periods=100, freq="B")
    x = pd.Series(np.linspace(1.0, 2.0, 100), index=index)
    y = 3.0 * x + 5.0
    beta = pd.Series(3.0, index=index)
    spread = relative_value.spread_returns(y, x, beta)
    assert spread.dropna().abs().max() == pytest.approx(0.0, abs=1e-12)


def test_every_backtested_pair_has_a_return_scale():
    for pair in relative_value.PAIRS:
        if pair.name in relative_value.BACKTEST_PAIRS:
            assert pair.return_scale is not None
        else:
            assert pair.return_scale is None


def test_primary_pair_is_declared_and_documented():
    pair = next(p for p in relative_value.PAIRS if p.name == relative_value.PRIMARY_PAIR)
    assert len(pair.rationale) > 100, "the primary pair needs a stated economic prior"
    assert pair.name in relative_value.BACKTEST_PAIRS


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------
def test_signal_registry_is_complete_and_unique():
    names = [s.name for s in signals.SIGNAL_REGISTRY]
    assert len(names) == len(set(names))
    hypotheses = {s.hypothesis for s in signals.SIGNAL_REGISTRY}
    assert len(hypotheses) == 3, "every hypothesis must contribute at least one candidate signal"
    assert all(s.description for s in signals.SIGNAL_REGISTRY)


def test_positioning_buckets_cover_the_declared_quantiles(feature_frame, panel):
    proxies = pd.concat([feature_frame.filter(like="ret_1d_"), features.return_proxies(panel)], axis=1).rename(
        columns=lambda c: c.removeprefix("ret_1d_")
    )
    study = signals.positioning_bucket_study(feature_frame, proxies, horizons=(5,))
    assert set(study["bucket"]) == {"bottom 10%", "10-25%", "25-75%", "75-90%", "top 10%"}
    assert (study["nobs"] > 30).all()
    assert study["market"].nunique() == len(config.COT_UNIVERSE)
