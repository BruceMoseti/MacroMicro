"""Pipeline orchestration.

    load -> validate -> features -> research -> signals -> backtest -> report

One function produces every number in the project. It returns a dict of tables so the
notebooks and the reporting layer read the same objects the validation checks ran
against, rather than recomputing anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import (
    backtest,
    charts,
    config,
    data_loader,
    features,
    metrics,
    regimes,
    regressions,
    relative_value,
    reporting,
    signals,
    validation,
)

ROLLING_RELATIONSHIP_PAIRS = {
    "equity_vs_rates": ("ret_1d_spx", "dy_1d_dgs10"),
    "equity_vs_vix": ("ret_1d_spx", "ret_1d_vix"),
    "usd_vs_wti": ("ret_1d_usd", "ret_1d_wti"),
    "gold_vs_real": ("ret_1d_gold", "dy_1d_real10"),
    "oil_vs_breakeven": ("ret_1d_wti", "dy_1d_breakeven10"),
    "rates_vs_usd": ("dy_1d_dgs10", "ret_1d_usd"),
}

# The single headline strategy, named before performance is inspected: the pre-specified
# primary relative-value pair, traded in its correct (hedged) form.
PRIMARY_STRATEGY = f"h2_rv_{relative_value.PRIMARY_PAIR}_hedged"


def _return_library(
    feature_frame: pd.DataFrame, hedged: pd.DataFrame, proxy_returns: pd.DataFrame
) -> dict[str, pd.Series]:
    """Every return series a registry entry can name, in one lookup."""
    library = {c: feature_frame[c] for c in feature_frame.columns if c.startswith("ret_1d_")}
    library.update({f"hedged_{c}": hedged[c] for c in hedged.columns})
    library.update({c: proxy_returns[c] for c in proxy_returns.columns})
    return library


def run(source: str = config.SOURCE_SYNTHETIC, write_outputs: bool = True) -> dict:
    log = validation.ValidationLog()
    results: dict[str, object] = {}

    # ---- 1. Load and validate the raw layer ------------------------------------------
    dataset = data_loader.load(source)
    panel, cot, vintages = dataset.daily, dataset.cot, dataset.macro_vintages
    validation.check_daily_panel(log, panel)
    validation.check_anchor_values(log, panel, source)
    validation.check_macro_vintages(log, vintages)
    validation.check_cot(log, cot)
    validation.check_futures_rolls(log)

    # ---- 2. Features -----------------------------------------------------------------
    curve_frame = features.curves(panel)
    feature_frame = features.build_full_features(panel, cot, vintages)
    validation.check_causality(log, "market_features", features.build_feature_frame, panel)
    validation.check_causality(log, "cot_features", lambda p: features.cot_features(cot, p.index), panel)
    validation.check_returns_finite(log, "features", feature_frame.filter(like="ret_1d_"))
    regime_frame = regimes.classify(panel, curve_frame)

    # ---- 3. Hypothesis 1: rates and equity risk --------------------------------------
    coefficients, diagnostics = regressions.run_specifications(feature_frame, "full")
    split_coefficients = pd.concat(
        [regressions.run_specifications(feature_frame, split)[0] for split in config.SPLITS], ignore_index=True
    )
    rolling_beta_frame = regressions.rolling_betas(feature_frame)
    stability = regressions.coefficient_stability(rolling_beta_frame)
    regime_coefficients = regressions.regime_regressions(feature_frame, regime_frame)
    oos = regressions.out_of_sample(feature_frame)
    validation.check_regression_reports_n(log, pd.concat([coefficients, split_coefficients], ignore_index=True))

    # ---- 4. Hypothesis 2: relative value ---------------------------------------------
    rv = relative_value.run(panel, curve_frame)
    primary_pair = next(p for p in relative_value.PAIRS if p.name == relative_value.PRIMARY_PAIR)
    convergence = relative_value.convergence_study(
        rv["residuals"][primary_pair.name],
        rv["zscores"][primary_pair.name],
        feature_frame[f"ret_1d_{primary_pair.tradable_leg}"],
    )

    # ---- 5. Hypothesis 3: positioning ------------------------------------------------
    proxy_returns = pd.concat(
        [feature_frame.filter(like="ret_1d_"), features.return_proxies(panel)], axis=1
    ).rename(columns=lambda c: c.removeprefix("ret_1d_"))
    bucket_study = signals.positioning_bucket_study(feature_frame, proxy_returns)
    extremes = signals.positioning_extremes_test(bucket_study)

    # ---- 6. Signals ------------------------------------------------------------------
    rv_states = signals.relative_value_positions(rv["zscores"])
    positioning_states = signals.positioning_positions(feature_frame)
    equity_states = signals.equity_prediction_positions(
        feature_frame, regressions.SPECIFICATIONS[regressions.PRIMARY_SPEC]
    )

    state_by_signal: dict[str, pd.Series] = {"h1_equity_forecast": equity_states}
    for pair in relative_value.PAIRS:
        if pair.return_scale is not None:
            state_by_signal[f"h2_rv_{pair.name}_hedged"] = rv_states[pair.name]
        if pair.name in relative_value.SINGLE_LEG_PAIRS:
            state_by_signal[f"h2_rv_{pair.name}_single"] = rv_states[pair.name]
    for market in config.COT_UNIVERSE:
        if market.name in positioning_states:
            state_by_signal[f"h3_positioning_{market.name}"] = positioning_states[market.name]

    # ---- 7. Backtests ----------------------------------------------------------------
    # Financing applies only to outright positions in a cash asset. Hedged spreads and
    # futures-proxy returns are already excess returns.
    financed = {
        spec.name
        for spec in signals.SIGNAL_REGISTRY
        if spec.name == "h1_equity_forecast" or spec.name.endswith("_single")
    }
    return_library = _return_library(feature_frame, rv["hedged_returns"], proxy_returns)
    # Beta is measured against the excess return of a long S&P 500 position, so that a
    # signal which is simply long most of the time cannot pass its beta off as alpha.
    equity_benchmark = feature_frame["ret_1d_spx"] - panel["fedfunds"] / 100.0 / config.TRADING_DAYS_PER_YEAR
    summary_rows, results_by_signal, position_by_signal = [], {}, {}
    for spec in signals.SIGNAL_REGISTRY:
        if not spec.tradable or spec.name not in state_by_signal:
            continue
        asset = return_library[spec.return_column]
        states = state_by_signal[spec.name]
        position = backtest.size_positions(states, asset)
        financing = panel["fedfunds"] if spec.name in financed else None
        position_by_signal[spec.name] = position
        summary_rows.append(
            backtest.cost_sensitivity(
                spec.name, position, asset, financing_pct=financing, benchmark=equity_benchmark
            )
        )
        results_by_signal[spec.name] = backtest.run(
            spec.name, position, asset, cost_bps=config.HEADLINE_COST_BPS, financing_pct=financing
        )
        validation.check_position_timing(log, spec.name, position, asset)
        validation.check_weights_known_in_advance(log, spec.name, position)

    backtest_summary = pd.concat(summary_rows, ignore_index=True)
    validation.check_backtest_reports_trades(log, backtest_summary)
    validation.check_causality(
        log,
        "relative_value_signal",
        lambda p: signals.relative_value_positions(relative_value.run(p, features.curves(p))["zscores"]),
        panel,
    )

    headline = backtest_summary[backtest_summary["cost_bps"] == config.HEADLINE_COST_BPS]
    consistency = backtest.split_consistency(headline)
    regime_table = pd.concat(
        [backtest.regime_performance(result, regime_frame) for result in results_by_signal.values()],
        ignore_index=True,
    )

    # ---- 8. Multiple testing ---------------------------------------------------------
    family = _test_family(coefficients, extremes, headline)
    holm = regressions.holm_bonferroni(family)

    # ---- 9. Rolling analysis ---------------------------------------------------------
    rolling_relationships = metrics.rolling_relationships(feature_frame, ROLLING_RELATIONSHIP_PAIRS)
    primary_result = results_by_signal[PRIMARY_STRATEGY]
    rolling_primary = metrics.rolling_analysis(primary_result.returns, benchmark=feature_frame["ret_1d_spx"])

    results.update(
        {
            "dataset": dataset,
            "panel": panel,
            "curves": curve_frame,
            "features": feature_frame,
            "regimes": regime_frame,
            "regime_summary": regimes.summary(regime_frame),
            "h1_coefficients": coefficients,
            "h1_split_coefficients": split_coefficients,
            "h1_diagnostics": diagnostics,
            "h1_rolling_betas": rolling_beta_frame,
            "h1_stability": stability,
            "h1_regime_coefficients": regime_coefficients,
            "h1_out_of_sample": oos,
            "rv_relationships": rv["relationships"],
            "rv_statistics": rv["statistics"],
            "rv_residuals": rv["residuals"],
            "rv_zscores": rv["zscores"],
            "rv_betas": rv["betas"],
            "rv_hedged_returns": rv["hedged_returns"],
            "rv_convergence": convergence,
            "positioning_buckets": bucket_study,
            "positioning_extremes": extremes,
            "signal_registry": signals.registry_frame(),
            "backtest_summary": backtest_summary,
            "backtest_headline": headline,
            "backtest_consistency": consistency,
            "backtest_results": results_by_signal,
            "positions": pd.DataFrame(position_by_signal),
            "regime_performance": regime_table,
            "holm": holm,
            "rolling_relationships": rolling_relationships,
            "rolling_primary": rolling_primary,
            "primary_strategy": PRIMARY_STRATEGY,
            "proxy_returns": proxy_returns,
        }
    )

    # ---- 10. Reporting ---------------------------------------------------------------
    if write_outputs:
        charts.reset_manifest()
        _build_charts(results, source)
        chart_manifest = charts.manifest()
        validation.check_charts_dated(log, chart_manifest)
        results["chart_manifest"] = chart_manifest

    results["validation"] = log.frame()
    results["validation_log"] = log
    results["interview_numbers"] = reporting.interview_numbers(results)

    if write_outputs:
        reporting.write_tables(results)
        reporting.write_excel_monitor(results)
        reporting.write_research_report(results)
        reporting.write_interview_sheet(results)
        panel.to_csv(config.DATA_PROCESSED / "daily_panel.csv")
        feature_frame.to_csv(config.DATA_PROCESSED / "features.csv")
        regime_frame.to_csv(config.DATA_PROCESSED / "regimes.csv")

    log.assert_ok()
    return results


def _test_family(coefficients: pd.DataFrame, extremes: pd.DataFrame, headline: pd.DataFrame) -> pd.Series:
    """Collect the project's headline p-values into one family for the Holm adjustment."""
    entries: dict[str, float] = {}
    primary = coefficients[(coefficients["spec"] == regressions.PRIMARY_SPEC) & (coefficients["term"] != "const")]
    for row in primary.itertuples():
        entries[f"H1 {row.term}"] = float(row.p_value)
    for row in extremes.itertuples():
        if row.horizon_days == 21:
            entries[f"H3 {row.market} top-vs-bottom decile 21d"] = _two_sided_p(row.t_stat)
    for row in headline[headline["split"] == "test"].itertuples():
        # Sharpe of an n-year sample has approximate standard error sqrt(1/n).
        years = max(row.years, 1e-6)
        entries[f"H2/H3 {row.signal} OOS Sharpe"] = _two_sided_p(row.sharpe * np.sqrt(years))
    return pd.Series(entries)


def _two_sided_p(t_stat: float) -> float:
    from scipy import stats

    if not np.isfinite(t_stat):
        return 1.0
    return float(2.0 * (1.0 - stats.norm.cdf(abs(t_stat))))


def _build_charts(results: dict, source: str) -> None:
    panel = results["panel"]
    charts.cross_asset_overview(panel, source)
    charts.yield_curve(panel, results["curves"], source)
    charts.real_yield_decomposition(panel, source)
    charts.correlation_matrix(results["features"], source)
    charts.rolling_correlations(results["rolling_relationships"], source)
    charts.rolling_equity_betas(results["h1_rolling_betas"], source)
    charts.levels_versus_changes(panel, results["rv_relationships"], source)
    charts.residual_zscore(results["rv_zscores"], results["rv_residuals"], relative_value.PRIMARY_PAIR, source)
    charts.rolling_hedge_ratios(results["rv_betas"], source)
    charts.positioning_history(results["features"], "zn", "10y Treasury note, leveraged funds", source)
    for horizon in (5, 21):
        charts.positioning_buckets(results["positioning_buckets"], horizon, source, panel.index)

    hypothesis_curves = {
        name: result.returns
        for name, result in results["backtest_results"].items()
        if name.endswith("_hedged") or name == "h1_equity_forecast"
    }
    charts.equity_curves(
        hypothesis_curves,
        source,
        "12_equity_curves.png",
        "Strategy equity curves: relative-value spreads and the predictive equity model",
    )
    charts.equity_curves(
        {name: r.returns for name, r in results["backtest_results"].items() if name.startswith("h3_")},
        source,
        "12b_equity_curves_positioning.png",
        "Strategy equity curves: futures positioning signals",
    )
    charts.cost_sensitivity(results["backtest_summary"], source, panel.index)
    charts.regime_bars(results["regime_performance"], results["primary_strategy"], source, panel.index)
    charts.rolling_performance(results["rolling_primary"], results["primary_strategy"], source)
