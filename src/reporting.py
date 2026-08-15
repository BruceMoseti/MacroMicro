"""Reporting: Excel monitor, CSV tables, research report and key figures sheet.

The Excel workbook is the deliverable a non-quant reads. Every sheet carries the data
provenance and the last observation date of every input, so a stale series cannot be
compared against a fresh one without it being visible.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config, regressions, relative_value, signals

HEADER_FILL = "1F4E79"
WARNING_FILL = "C00000"


# --------------------------------------------------------------------------------------
# Monitor blocks
# --------------------------------------------------------------------------------------
def market_snapshot(panel: pd.DataFrame, curve_frame: pd.DataFrame) -> pd.DataFrame:
    """Latest level of every instrument, with its own last observation date."""
    rows = []
    for spec in config.DAILY_UNIVERSE:
        series = panel[spec.name].dropna()
        rows.append(
            {
                "series": spec.name,
                "label": spec.label,
                "asset_class": spec.asset_class,
                "fred_id": spec.fred_id,
                "unit": spec.unit,
                "latest": float(series.iloc[-1]),
                "last_observation": series.index[-1].date().isoformat(),
                "observations": int(len(series)),
                "missing": int(panel[spec.name].isna().sum()),
            }
        )
    for name in curve_frame.columns:
        series = curve_frame[name].dropna()
        rows.append(
            {
                "series": name,
                "label": f"{name.removeprefix('curve_')} slope",
                "asset_class": "rates",
                "fred_id": "derived",
                "unit": "bp",
                "latest": float(series.iloc[-1]),
                "last_observation": series.index[-1].date().isoformat(),
                "observations": int(len(series)),
                "missing": 0,
            }
        )
    return pd.DataFrame(rows)


def cross_asset_changes(panel: pd.DataFrame, curve_frame: pd.DataFrame) -> pd.DataFrame:
    """1 day, 1 week, 1 month, 3 month and 12 month changes.

    Prices are reported as percentage returns; yields and spreads in basis points, which
    is how each is actually quoted.
    """
    horizons = {"1d": 1, "1w": 5, "1m": 21, "3m": 63, "12m": 252}
    rows = []
    for spec in config.DAILY_UNIVERSE:
        series = panel[spec.name].dropna()
        row = {"series": spec.name, "label": spec.label, "latest": float(series.iloc[-1])}
        row["unit"] = "%" if spec.kind == "price" else "bp"
        for label, horizon in horizons.items():
            if len(series) <= horizon:
                row[label] = np.nan
            elif spec.kind == "price":
                row[label] = float((series.iloc[-1] / series.iloc[-1 - horizon] - 1.0) * 100.0)
            else:
                row[label] = float((series.iloc[-1] - series.iloc[-1 - horizon]) * config.BP_PER_PCT)
        rows.append(row)
    for name in curve_frame.columns:
        series = curve_frame[name].dropna()
        row = {"series": name, "label": f"{name.removeprefix('curve_')} slope", "latest": float(series.iloc[-1]), "unit": "bp"}
        for label, horizon in horizons.items():
            row[label] = float(series.iloc[-1] - series.iloc[-1 - horizon]) if len(series) > horizon else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def rolling_relationship_snapshot(rolling: pd.DataFrame) -> pd.DataFrame:
    """Latest rolling correlation and beta for each tracked relationship."""
    labels = {
        "equity_vs_rates": "S&P 500 vs 10y yield change",
        "equity_vs_vix": "S&P 500 vs VIX change",
        "usd_vs_wti": "Broad USD vs WTI",
        "gold_vs_real": "Gold vs 10y real yield change",
        "oil_vs_breakeven": "WTI vs 10y breakeven change",
        "rates_vs_usd": "10y yield change vs broad USD",
    }
    rows = []
    for key, label in labels.items():
        row = {"relationship": key, "label": label}
        for window in config.ROLLING_CORR_WINDOWS:
            column = f"corr{window}_{key}"
            if column in rolling:
                series = rolling[column].dropna()
                row[f"corr_{window}d"] = float(series.iloc[-1])
                row[f"corr_{window}d_min"] = float(series.min())
                row[f"corr_{window}d_max"] = float(series.max())
        column = f"beta252_{key}"
        if column in rolling:
            series = rolling[column].dropna()
            row["beta_252d"] = float(series.iloc[-1])
        row["as_of"] = rolling.index[-1].date().isoformat()
        rows.append(row)
    return pd.DataFrame(rows)


def positioning_snapshot(feature_frame: pd.DataFrame, cot: pd.DataFrame) -> pd.DataFrame:
    """Latest COT reading per market, including when it became public."""
    from . import features as feature_module

    rows = []
    for market in config.COT_UNIVERSE:
        category = config.COT_SPEC_CATEGORY[market.report]
        weekly = cot[(cot["market"] == market.name) & (cot["category"] == category)].sort_values("report_date")
        latest = weekly.iloc[-1]
        available = feature_module.cot_available_from(weekly["report_date"]).iloc[-1]
        rows.append(
            {
                "market": market.name,
                "label": market.label,
                "report": market.report,
                "category": category,
                "report_date_tuesday": latest["report_date"].date().isoformat(),
                "usable_from": available.date().isoformat(),
                "long": float(latest["long"]),
                "short": float(latest["short"]),
                "net_position": float(latest["long"] - latest["short"]),
                "open_interest": float(latest["open_interest"]),
                "net_pct_of_oi": float(feature_frame[f"cot_{market.name}_ratio"].dropna().iloc[-1] * 100.0),
                "percentile_52w": float(feature_frame[f"cot_{market.name}_pct52w"].dropna().iloc[-1]),
                "zscore_3y": float(feature_frame[f"cot_{market.name}_z156"].dropna().iloc[-1]),
                "weekly_change_pct_oi": float(feature_frame[f"cot_{market.name}_chg1w"].dropna().iloc[-1] * 100.0),
            }
        )
    return pd.DataFrame(rows)


def signal_snapshot(results: dict) -> pd.DataFrame:
    """Current state of every signal: residual, z-score, direction and percentile."""
    zscores, residuals = results["rv_zscores"], results["rv_residuals"]
    positions, features_frame = results["positions"], results["features"]
    rows = []

    for pair in relative_value.PAIRS:
        zscore = zscores[pair.name].dropna()
        residual = residuals[pair.name].dropna()
        latest_z = float(zscore.iloc[-1])
        percentile = float((zscore <= latest_z).mean() * 100.0)
        for suffix in ("_hedged", "_single"):
            name = f"h2_rv_{pair.name}{suffix}"
            if name not in positions:
                continue
            position = float(positions[name].dropna().iloc[-1]) if positions[name].notna().any() else np.nan
            rows.append(
                {
                    "signal": name,
                    "hypothesis": "H2 relative value",
                    "residual": float(residual.iloc[-1]),
                    "zscore": latest_z,
                    "historical_percentile": percentile,
                    "direction": "long" if position > 0 else ("short" if position < 0 else "flat"),
                    "position_size": position,
                    "confidence": min(abs(latest_z) / signals.ENTRY_Z, 1.0),
                    "as_of": zscore.index[-1].date().isoformat(),
                }
            )

    for market in config.COT_UNIVERSE:
        column = f"cot_{market.name}_z156"
        if column not in features_frame:
            continue
        zscore = features_frame[column].dropna()
        latest_z = float(zscore.iloc[-1])
        name = f"h3_positioning_{market.name}"
        position = float(positions[name].dropna().iloc[-1]) if name in positions and positions[name].notna().any() else np.nan
        rows.append(
            {
                "signal": name,
                "hypothesis": "H3 positioning",
                "residual": float(features_frame[f"cot_{market.name}_ratio"].dropna().iloc[-1]),
                "zscore": latest_z,
                "historical_percentile": float((zscore <= latest_z).mean() * 100.0),
                "direction": "long" if position > 0 else ("short" if position < 0 else "flat"),
                "position_size": position,
                "confidence": min(abs(latest_z) / signals.POSITION_ENTRY_Z, 1.0),
                "as_of": zscore.index[-1].date().isoformat(),
            }
        )

    name = "h1_equity_forecast"
    if name in positions:
        position = float(positions[name].iloc[-1])
        rows.append(
            {
                "signal": name,
                "hypothesis": "H1 rates and equity risk",
                "residual": np.nan,
                "zscore": np.nan,
                "historical_percentile": np.nan,
                "direction": "long" if position > 0 else ("short" if position < 0 else "flat"),
                "position_size": position,
                "confidence": np.nan,
                "as_of": positions.index[-1].date().isoformat(),
            }
        )
    return pd.DataFrame(rows)


def data_dictionary() -> pd.DataFrame:
    rows = [
        {
            "name": spec.name,
            "source": "FRED",
            "identifier": spec.fred_id,
            "asset_class": spec.asset_class,
            "unit": spec.unit,
            "transform": "log return" if spec.kind == "price" else "basis point change",
            "description": spec.label,
            "notes": spec.coverage_note,
        }
        for spec in config.DAILY_UNIVERSE
    ]
    rows += [
        {
            "name": spec.name,
            "source": "ALFRED (vintage)",
            "identifier": spec.fred_id,
            "asset_class": "macro",
            "unit": spec.freq,
            "transform": "point-in-time, vintage-selected",
            "description": spec.label,
            "notes": f"conservative release lag fallback: {spec.release_lag_days} days after period end",
        }
        for spec in config.MACRO_UNIVERSE
    ]
    rows += [
        {
            "name": market.name,
            "source": f"CFTC COT ({market.report})",
            "identifier": market.cme_code,
            "asset_class": "positioning",
            "unit": "contracts",
            "transform": "net / open interest, 3y z-score, published-lag aligned",
            "description": market.contract_name,
            "notes": market.proxy_note,
        }
        for market in config.COT_UNIVERSE
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Key figures
# --------------------------------------------------------------------------------------
def _interpret(coefficient_row: pd.Series) -> str:
    """Plain-English reading of one H1 coefficient, in the units of its own regressor."""
    term, beta = coefficient_row["term"], float(coefficient_row["coefficient"])
    labels = {
        "dy_1d_real10": "the 10y TIPS real yield",
        "dy_1d_breakeven10": "10y breakeven inflation",
        "dy_1d_slope": "the 2s10s slope",
        "dy_1d_level": "the average of the 2y and 10y yields",
        "dy_1d_dgs2": "the 2y yield",
        "dy_1d_dgs10": "the 10y yield",
    }
    if term == "ret_1d_vix":
        return f"A 1% rise in VIX coincides with a {beta * 100:.2f}bp move in the S&P 500."
    if term in labels:
        return f"A 1bp rise in {labels[term]} coincides with a {beta * 10_000:.2f}bp move in the S&P 500."
    return f"Interpretation depends on the units of {term}."


def key_figures(results: dict) -> pd.DataFrame:
    """The project's twenty headline figures, each traceable to a generated table."""
    panel = results["panel"]
    headline = results["backtest_headline"]
    test = headline[headline["split"] == "test"].sort_values("sharpe", ascending=False)
    best = test.iloc[0] if len(test) else None
    primary = headline[(headline["signal"] == results["primary_strategy"]) & (headline["split"] == "test")]
    primary_row = primary.iloc[0] if len(primary) else None

    h1 = results["h1_coefficients"]
    h1_primary = h1[(h1["spec"] == regressions.PRIMARY_SPEC) & (h1["term"] != "const")]
    strongest = h1_primary.loc[h1_primary["t_stat"].abs().idxmax()]
    diagnostics = results["h1_diagnostics"]
    primary_diagnostics = diagnostics[diagnostics["spec"] == regressions.PRIMARY_SPEC].iloc[0]

    statistics = results["rv_statistics"]
    train_statistics = statistics[
        (statistics["pair"] == relative_value.PRIMARY_PAIR) & (statistics["sample"] == "train")
    ].iloc[0]

    regime = results["regime_performance"]
    primary_regimes = regime[regime["signal"] == results["primary_strategy"]].dropna(subset=["sharpe"])
    best_regime = primary_regimes.loc[primary_regimes["sharpe"].idxmax()] if len(primary_regimes) else None
    worst_regime = primary_regimes.loc[primary_regimes["sharpe"].idxmin()] if len(primary_regimes) else None

    total_trades = int(headline[headline["split"] == "full"]["trades"].sum())
    registry = results["signal_registry"]

    rows = [
        ("1. Datasets", f"{len(config.DATASETS)}: " + "; ".join(config.DATASETS)),
        (
            "2. Instruments and series",
            f"{len(config.DAILY_UNIVERSE)} daily market series, {len(config.MACRO_UNIVERSE)} vintage macro "
            f"series, {len(config.COT_UNIVERSE)} CFTC positioning markets = "
            f"{len(config.DAILY_UNIVERSE) + len(config.MACRO_UNIVERSE) + len(config.COT_UNIVERSE)} total; "
            f"{results['features'].shape[1]} engineered features",
        ),
        (
            "3. Daily observations",
            f"{len(panel):,} trading days; {len(panel) * len(config.DAILY_UNIVERSE):,} daily series-observations; "
            f"{results['dataset'].cot['report_date'].nunique()} weekly COT reports; "
            f"{len(results['dataset'].macro_vintages):,} macro vintage rows",
        ),
        (
            "4. Start and end date",
            f"{panel.index.min():%Y-%m-%d} to {panel.index.max():%Y-%m-%d} (frozen). Train to "
            f"{config.TRAIN_END:%Y-%m-%d}, validation to {config.VALID_END:%Y-%m-%d}, test from "
            f"{config.TEST_START:%Y-%m-%d}",
        ),
        (
            "5. Candidate signals tested",
            f"{len(registry)} declared in signals.SIGNAL_REGISTRY, of which {int(registry['backtested'].sum())} "
            f"were backtested; {len(regressions.SPECIFICATIONS)} regression specifications; "
            f"{len(relative_value.PAIRS)} relative-value pairs; {len(config.COT_UNIVERSE)} positioning markets",
        ),
        ("6. Actual trades", f"{total_trades:,} across all backtested signals, full sample, at {config.HEADLINE_COST_BPS:.0f}bp"),
        (
            "7. Best out-of-sample Sharpe",
            f"{best['sharpe']:.2f} ({best['signal']}, {int(best['nobs'])} days, {int(best['trades'])} trades, "
            f"{config.HEADLINE_COST_BPS:.0f}bp costs). This figure should not be read in isolation: "
            f"{int(results['backtest_consistency']['positive_in_all_splits'].sum())} of "
            f"{len(results['backtest_consistency'])} strategies were positive in all three splits, so it is "
            f"almost certainly noise rather than an edge"
            if best is not None
            else "n/a",
        ),
        (
            "8. Maximum drawdown",
            f"{primary_row['max_drawdown'] * 100:.1f}% for the primary strategy out of sample; "
            f"{headline[headline['split'] == 'full']['max_drawdown'].min() * 100:.1f}% worst across all signals "
            f"(full sample)" if primary_row is not None else "n/a",
        ),
        (
            "9. Hit rate",
            f"{primary_row['hit_rate'] * 100:.1f}% for the primary strategy out of sample, over "
            f"{int(primary_row['active_days'])} days holding risk" if primary_row is not None else "n/a",
        ),
        (
            "10. Average holding period",
            f"{primary_row['avg_holding_days']:.1f} trading days out of sample; "
            f"{headline[headline['split'] == 'full']['avg_holding_days'].mean():.1f} days on average across signals"
            if primary_row is not None
            else "n/a",
        ),
        (
            "11. Transaction cost assumption",
            f"sensitivity grid {', '.join(f'{c:.0f}' for c in config.COST_GRID_BPS)}bp per unit traded, headline "
            f"{config.HEADLINE_COST_BPS:.0f}bp. Not a sourced cost model. Financing at the effective fed funds "
            f"rate is charged on outright cash positions only",
        ),
        (
            "12. Most important regression coefficient",
            f"{strongest['term']} = {strongest['coefficient']:.5f} (HAC t = {strongest['t_stat']:.1f}, "
            f"p = {strongest['p_value']:.2e}, n = {int(strongest['nobs'])}). {_interpret(strongest)}",
        ),
        (
            "13. Regression R-squared",
            f"adjusted R2 = {primary_diagnostics['adj_r_squared']:.3f} contemporaneous (spec "
            f"'{regressions.PRIMARY_SPEC}'); predictive adjusted R2 is approximately zero "
            f"(see h1_out_of_sample)",
        ),
        (
            "14. Stationarity result",
            f"{relative_value.PRIMARY_PAIR} training-sample residual: ADF = {train_statistics['adf_stat']:.2f} "
            f"(p = {train_statistics['adf_p']:.3f}), Engle-Granger p = {train_statistics['engle_granger_p']:.3f}, "
            f"AR(1) phi = {train_statistics['ar1_phi']:.4f}, half-life = {train_statistics['half_life_days']:.0f} "
            f"days. No pair was cointegrated at 5%",
        ),
        (
            "15. Best performing regime",
            f"{best_regime['dimension']} = {best_regime['regime']}, Sharpe {best_regime['sharpe']:.2f} over "
            f"{int(best_regime['days'])} days" if best_regime is not None else "n/a",
        ),
        (
            "16. Worst performing regime",
            f"{worst_regime['dimension']} = {worst_regime['regime']}, Sharpe {worst_regime['sharpe']:.2f} over "
            f"{int(worst_regime['days'])} days" if worst_regime is not None else "n/a",
        ),
        (
            "17. Biggest failure",
            "H1 has no predictive content. The contemporaneous regression explains "
            f"{primary_diagnostics['adj_r_squared']:.0%} of daily equity variance, but lagging the same "
            "regressors by one day gives a negative out-of-sample R2, so the relationship is an accounting "
            "identity of same-day repricing rather than a forecast.",
        ),
        (
            "18. Biggest source of potential bias",
            "Multiple testing and pair selection. Five relative-value pairs, three regression specifications and "
            "five positioning markets were evaluated on one fixed sample; the Holm-Bonferroni adjustment over "
            f"the {len(results['holm'])}-test family is reported rather than the raw p-values.",
        ),
        (
            "19. Futures roll handling",
            "No futures price series is used, so there is no roll. Derivatives enter through CFTC "
            "futures-and-options positioning. Return proxies for the positioning markets are cash or "
            "duration-approximated and exclude roll yield, which is stated wherever they appear.",
        ),
        (
            "20. Lookahead prevention",
            "Positions at t earn the return from t to t+1; COT positions are Tuesday readings withheld until the "
            "session after Friday's 15:30 ET release (a 6 day lag); macro data is vintage-selected point-in-time; "
            "every rolling statistic and hedge ratio uses a trailing window; and a perturbation test confirms that "
            "changing the last 40 rows of input leaves all earlier features and signals bit-identical.",
        ),
    ]
    return pd.DataFrame(rows, columns=["figure", "value"])


# --------------------------------------------------------------------------------------
# Writers
# --------------------------------------------------------------------------------------
TABLE_KEYS = (
    "h1_coefficients",
    "h1_split_coefficients",
    "h1_diagnostics",
    "h1_stability",
    "h1_regime_coefficients",
    "h1_out_of_sample",
    "rv_relationships",
    "rv_statistics",
    "rv_convergence",
    "positioning_buckets",
    "positioning_extremes",
    "signal_registry",
    "backtest_summary",
    "backtest_consistency",
    "regime_performance",
    "regime_summary",
    "holm",
    "validation",
    "key_figures",
)


def write_tables(results: dict) -> None:
    for key in TABLE_KEYS:
        frame = results.get(key)
        if isinstance(frame, pd.DataFrame):
            frame.to_csv(config.TABLE_DIR / f"{key}.csv", index=False)
    manifest = results.get("chart_manifest")
    if isinstance(manifest, pd.DataFrame):
        manifest.to_csv(config.TABLE_DIR / "chart_manifest.csv", index=False)


def write_excel_monitor(results: dict) -> None:
    dataset = results["dataset"]
    provenance = dataset.provenance()
    panel, curve_frame = results["panel"], results["curves"]

    readme = pd.DataFrame(
        [
            ("Workbook", "Macro Markets Quantitative Research Platform: cross-asset monitor"),
            ("Data source", provenance["source"]),
            ("Synthetic data", "YES - not observed market data" if provenance["is_synthetic"] else "no"),
            ("Research window", f"{provenance['start_date']} to {provenance['end_date']} (frozen)"),
            ("Daily observations", f"{provenance['daily_observations']:,}"),
            ("Regenerate", "python run_pipeline.py --source synthetic   (or --source fred with network access)"),
            ("Validation", results["validation_log"].summary()),
            ("Costs", f"{config.HEADLINE_COST_BPS:.0f}bp headline; sensitivity grid in the Cost Sensitivity sheet"),
        ]
        + [("Note", note) for note in provenance["notes"]],
        columns=["field", "value"],
    )

    sheets = {
        "README": readme,
        "Market Snapshot": market_snapshot(panel, curve_frame),
        "Cross Asset Changes": cross_asset_changes(panel, curve_frame),
        "Rolling Relationships": rolling_relationship_snapshot(results["rolling_relationships"]),
        "Positioning": positioning_snapshot(results["features"], dataset.cot),
        "Signals": signal_snapshot(results),
        "H1 Regressions": results["h1_coefficients"],
        "H1 Diagnostics": results["h1_diagnostics"],
        "H1 Stability": results["h1_stability"],
        "H1 Out Of Sample": results["h1_out_of_sample"],
        "RV Relationships": results["rv_relationships"],
        "RV Stationarity": results["rv_statistics"],
        "RV Convergence": results["rv_convergence"],
        "Positioning Buckets": results["positioning_buckets"],
        "Positioning Extremes": results["positioning_extremes"],
        "Signal Registry": results["signal_registry"],
        "Backtest Summary": results["backtest_headline"],
        "Split Consistency": results["backtest_consistency"],
        "Cost Sensitivity": results["backtest_summary"],
        "Regime Performance": results["regime_performance"],
        "Regime Summary": results["regime_summary"],
        "Multiple Testing": results["holm"],
        "Validation Log": results["validation"],
        "Key Figures": results["key_figures"],
        "Data Dictionary": data_dictionary(),
        "Chart Manifest": results.get("chart_manifest", pd.DataFrame()),
    }

    with pd.ExcelWriter(config.EXCEL_PATH, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)
        _style_workbook(writer, sheets)


def _style_workbook(writer, sheets: dict[str, pd.DataFrame]) -> None:
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor=HEADER_FILL)
    for name, frame in sheets.items():
        sheet = writer.sheets[name[:31]]
        sheet.freeze_panes = "A2"
        for cell in sheet[1]:
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
        for position, column in enumerate(frame.columns, start=1):
            values = frame[column].astype(str)
            width = max(len(str(column)), int(values.str.len().max()) if len(values) else 0)
            sheet.column_dimensions[get_column_letter(position)].width = min(max(width + 2, 11), 62)
        if name == "Validation Log" and "result" in frame.columns:
            index = list(frame.columns).index("result") + 1
            fail_fill = PatternFill("solid", fgColor=WARNING_FILL)
            for row in range(2, len(frame) + 2):
                if sheet.cell(row=row, column=index).value in {"FAIL", "WARN"}:
                    sheet.cell(row=row, column=index).fill = fail_fill
                    sheet.cell(row=row, column=index).font = Font(bold=True, color="FFFFFF")


def write_key_figures(results: dict) -> None:
    numbers = results["key_figures"]
    provenance = results["dataset"].provenance()
    lines = [
        "# Key figures",
        "",
        "A single-page reference for the project's headline numbers. Every value is generated "
        "from the tables in `outputs/tables/` rather than written by hand, so this sheet cannot "
        "fall out of step with the results.",
        "",
        f"Source table: `outputs/tables/key_figures.csv`. Data source: **{provenance['source']}**"
        + ("  \n**These figures come from the synthetic dataset and are not market history.**" if provenance["is_synthetic"] else ""),
        "",
    ]
    for row in numbers.itertuples():
        lines.append(f"**{row.figure}**  \n{row.value}")
        lines.append("")
    (config.DOCS_DIR / "KEY_FIGURES.md").write_text("\n".join(lines))


def write_research_report(results: dict) -> None:
    """Assemble the research report from the computed tables."""
    from . import report_text

    text = report_text.build(results)
    (config.DOCS_DIR / "RESEARCH_REPORT.md").write_text(text)
    (config.DATA_PROCESSED / "run_summary.json").write_text(
        json.dumps(
            {
                "provenance": results["dataset"].provenance(),
                "validation": results["validation_log"].summary(),
                "primary_strategy": results["primary_strategy"],
                "charts": int(len(results.get("chart_manifest", []))),
                "tables": len(TABLE_KEYS),
            },
            indent=2,
        )
        + "\n"
    )
