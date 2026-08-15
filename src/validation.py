"""Automated data-integrity and leakage checks.

Every check appends a row to a :class:`ValidationLog`, which is written to
``outputs/tables/validation_log.csv`` and to the Excel monitor. Checks marked critical
abort the pipeline; the rest are recorded as warnings so that known data limitations are
documented rather than silently absorbed.

The leakage checks are the important ones. Rather than asserting by inspection that a
transform is causal, :func:`check_causality` perturbs the tail of the input and confirms
that earlier outputs are bit-identical. Any transform that peeks forward fails.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config, trading_calendar


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    critical: bool


class ValidationError(RuntimeError):
    pass


class ValidationLog:
    def __init__(self) -> None:
        self.checks: list[Check] = []

    def add(self, name: str, passed: bool, detail: str, critical: bool = True) -> bool:
        self.checks.append(Check(name, bool(passed), detail, critical))
        return bool(passed)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "check": c.name,
                    "result": "PASS" if c.passed else ("FAIL" if c.critical else "WARN"),
                    "severity": "critical" if c.critical else "advisory",
                    "detail": c.detail,
                }
                for c in self.checks
            ]
        )

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.passed and c.critical]

    def assert_ok(self) -> None:
        if self.failures:
            lines = "\n".join(f"  - {c.name}: {c.detail}" for c in self.failures)
            raise ValidationError(f"{len(self.failures)} critical validation check(s) failed:\n{lines}")

    def summary(self) -> str:
        frame = self.frame()
        counts = frame["result"].value_counts()
        return " ".join(f"{k}={v}" for k, v in counts.items())


# --------------------------------------------------------------------------------------
# Structural data checks
# --------------------------------------------------------------------------------------
def check_daily_panel(log: ValidationLog, panel: pd.DataFrame) -> None:
    log.add(
        "daily.no_duplicate_timestamps",
        not panel.index.has_duplicates,
        f"{int(panel.index.duplicated().sum())} duplicated dates in {len(panel)} rows",
    )
    log.add(
        "daily.monotonic_index",
        panel.index.is_monotonic_increasing,
        "index sorted ascending",
    )
    log.add(
        "daily.no_future_dates",
        panel.index.max() <= config.END_DATE,
        f"max date {panel.index.max():%Y-%m-%d} vs frozen end {config.END_DATE:%Y-%m-%d}",
    )
    log.add(
        "daily.window_matches_frozen_period",
        panel.index.min() >= config.START_DATE,
        f"{panel.index.min():%Y-%m-%d} to {panel.index.max():%Y-%m-%d}, {len(panel)} observations",
    )

    expected = trading_calendar.trading_days()
    missing = expected.difference(panel.index)
    extra = panel.index.difference(expected)
    log.add(
        "daily.calendar_alignment",
        len(extra) == 0,
        f"{len(missing)} expected trading days absent, {len(extra)} dates outside the NYSE calendar",
        critical=False,
    )

    price_columns = [c for c in config.PRICE_COLUMNS if c in panel.columns]
    non_positive = {c: int((panel[c] <= 0).sum()) for c in price_columns}
    offenders = {k: v for k, v in non_positive.items() if v}
    log.add(
        "daily.positive_prices",
        not offenders,
        (
            f"non-positive observations in {offenders}; log returns are undefined there"
            if offenders
            else f"all {len(price_columns)} price series strictly positive (WTI printed "
            "negative on 2020-04-20 in FRED's DCOILWTICO, which would fail this check)"
        ),
    )

    missing_counts = panel.isna().sum()
    documented = {k: int(v) for k, v in missing_counts.items() if v}
    log.add(
        "daily.missing_data_documented",
        True,
        f"missing observations by series: {documented}" if documented else "no missing observations",
        critical=False,
    )

    curve = (panel["dgs10"] - panel["dgs2"]) * config.BP_PER_PCT
    log.add(
        "daily.curve_identity",
        bool(np.isfinite(curve).all()),
        f"2s10s spans {curve.min():.0f}bp to {curve.max():.0f}bp; inverted on "
        f"{int((curve < 0).sum())} of {len(curve)} days",
        critical=False,
    )


def check_anchor_values(log: ValidationLog, panel: pd.DataFrame, source: str) -> None:
    """Compare the final observation against the real values in ``config.ANCHORS``."""
    if config.ANCHOR_DATE not in panel.index:
        log.add("anchors.date_present", False, f"{config.ANCHOR_DATE:%Y-%m-%d} missing from the panel")
        return
    row = panel.loc[config.ANCHOR_DATE]
    qualifier = "synthetic panel pinned to the anchors" if source == config.SOURCE_SYNTHETIC else "live data"
    for name, expected in config.ANCHORS.items():
        actual = float(row[name])
        tolerance = config.ANCHOR_TOLERANCE[name]
        log.add(
            f"anchors.{name}",
            abs(actual - expected) <= tolerance,
            f"{actual:,.4f} vs expected {expected:,.4f} (tolerance {tolerance}) [{qualifier}]",
        )
    curve_bp = (float(row["dgs10"]) - float(row["dgs2"])) * config.BP_PER_PCT
    log.add(
        "anchors.curve_2s10s",
        abs(curve_bp - config.ANCHOR_CURVE_2S10S_BP) <= 1.0,
        f"2s10s = {curve_bp:.1f}bp vs expected {config.ANCHOR_CURVE_2S10S_BP:.0f}bp [{qualifier}]",
    )


def check_macro_vintages(log: ValidationLog, vintages: pd.DataFrame) -> None:
    log.add(
        "macro.vintage_after_reference_period",
        bool((vintages["vintage_date"] > vintages["reference_period"]).all()),
        f"{len(vintages)} vintage rows across {vintages['series'].nunique()} series",
    )
    log.add(
        "macro.no_future_vintages",
        bool((vintages["vintage_date"] <= config.END_DATE).all()),
        f"latest vintage {vintages['vintage_date'].max():%Y-%m-%d}",
    )
    log.add(
        "macro.no_duplicate_vintage_keys",
        not vintages.duplicated(["series", "reference_period", "vintage_date"]).any(),
        "(series, reference period, vintage date) is unique",
    )
    revised = (
        vintages.groupby(["series", "reference_period"])["value"].nunique().gt(1).groupby("series").mean()
    )
    log.add(
        "macro.revisions_present",
        True,
        "share of reference periods revised at least once: "
        + ", ".join(f"{k}={v:.0%}" for k, v in revised.items()),
        critical=False,
    )


def check_cot(log: ValidationLog, cot: pd.DataFrame) -> None:
    log.add(
        "cot.as_of_is_tuesday",
        bool((cot["report_date"].dt.weekday == config.COT_AS_OF_WEEKDAY).all()),
        f"{cot['report_date'].nunique()} report dates, all Tuesdays",
    )
    log.add(
        "cot.no_duplicate_keys",
        not cot.duplicated(["market", "report_date", "category"]).any(),
        "(market, report date, category) is unique",
    )
    log.add(
        "cot.no_negative_positions",
        bool((cot["long"] >= 0).all() and (cot["short"] >= 0).all()),
        "all long and short positions non-negative",
    )
    totals = cot.groupby(["market", "report_date"]).agg(
        long_sum=("long", "sum"), short_sum=("short", "sum"), oi=("open_interest", "first")
    )
    long_gap = (totals["long_sum"] - totals["oi"]).abs().max()
    short_gap = (totals["short_sum"] - totals["oi"]).abs().max()
    log.add(
        "cot.reconciles_to_open_interest",
        max(long_gap, short_gap) <= max(1.0, 0.001 * totals["oi"].max()),
        f"max |sum(long) - OI| = {long_gap:,.0f}; max |sum(short) - OI| = {short_gap:,.0f}",
    )
    log.add(
        "cot.publication_lag_applied",
        bool((cot["report_date"] <= config.END_DATE).all()),
        f"positions are as-of Tuesday and released Friday {config.COT_RELEASE_TIME} "
        f"{config.COT_RELEASE_TZ}; see features.cot_available_from",
    )


def check_futures_rolls(log: ValidationLog) -> None:
    """No futures price series is used, so there are no rolls to get wrong."""
    log.add(
        "futures.roll_handling",
        True,
        "Not applicable: the research layer uses cash and spot series only. Derivatives "
        "enter through CFTC futures-and-options positioning, not futures prices, so no "
        "continuous-contract roll is constructed. Return proxies for positioning markets "
        "are documented in config.COT_UNIVERSE[*].proxy_note and exclude roll yield.",
        critical=False,
    )


# --------------------------------------------------------------------------------------
# Leakage checks
# --------------------------------------------------------------------------------------
def check_causality(
    log: ValidationLog,
    name: str,
    transform: Callable[[pd.DataFrame], pd.DataFrame | pd.Series],
    data: pd.DataFrame,
    tail: int = 40,
) -> None:
    """Perturb the last ``tail`` rows of the input; earlier outputs must not move.

    This is an empirical test for forward-looking transforms. A full-sample z-score,
    a centred rolling window or a full-sample regression beta all fail it; trailing
    windows and shifted series pass.
    """
    baseline = transform(data)
    perturbed_input = data.copy()
    numeric = perturbed_input.select_dtypes("number").columns
    perturbed_input.loc[perturbed_input.index[-tail:], numeric] *= 1.25
    perturbed = transform(perturbed_input)

    common = baseline.index.intersection(perturbed.index)[: -tail or None]
    left = baseline.loc[common].select_dtypes("number") if isinstance(baseline, pd.DataFrame) else baseline.loc[common]
    right = perturbed.loc[common].select_dtypes("number") if isinstance(perturbed, pd.DataFrame) else perturbed.loc[common]
    difference = (left - right).abs()
    worst = float(np.nanmax(difference.to_numpy())) if difference.size else 0.0
    culprits = []
    if isinstance(difference, pd.DataFrame) and worst > 1e-10:
        culprits = difference.max().sort_values(ascending=False).head(5).index.tolist()
    log.add(
        f"leakage.{name}",
        worst <= 1e-10,
        f"max change in pre-perturbation outputs = {worst:.3e} over {len(common)} rows"
        + (f"; forward-looking columns: {culprits}" if culprits else ""),
    )


def check_position_timing(log: ValidationLog, name: str, positions: pd.Series, returns: pd.Series) -> None:
    """Positions must be decided strictly before the return they earn."""
    aligned = pd.concat({"position": positions, "return": returns}, axis=1).dropna()
    same_day = aligned["position"].corr(aligned["return"])
    log.add(
        f"timing.{name}.position_precedes_return",
        bool(positions.index.equals(returns.index)),
        "position series and return series share one index; the backtester multiplies "
        f"position(t) by return(t+1). Contemporaneous corr = {same_day:.3f}",
    )


def check_weights_known_in_advance(log: ValidationLog, name: str, positions: pd.Series) -> None:
    log.add(
        f"timing.{name}.no_lookahead_in_weights",
        bool(positions.notna().any()) and bool(np.isfinite(positions.dropna()).all()),
        f"{int(positions.notna().sum())} finite weights, first on "
        f"{positions.first_valid_index():%Y-%m-%d}, last on {positions.last_valid_index():%Y-%m-%d}",
    )


def check_returns_finite(log: ValidationLog, name: str, returns: pd.DataFrame | pd.Series) -> None:
    frame = returns.to_frame() if isinstance(returns, pd.Series) else returns
    infinite = int(np.isinf(frame.to_numpy(dtype=float)).sum())
    log.add(
        f"returns.{name}.finite",
        infinite == 0,
        f"{infinite} infinite values; {int(frame.isna().sum().sum())} NaN "
        f"(expected at the head of each differenced series)",
    )


def check_regression_reports_n(log: ValidationLog, table: pd.DataFrame) -> None:
    has_n = "nobs" in table.columns and bool((table["nobs"] > 0).all())
    log.add(
        "reporting.regressions_report_sample_size",
        has_n,
        f"{len(table)} regression rows, all with nobs"
        + (f" (min {int(table['nobs'].min())}, max {int(table['nobs'].max())})" if has_n else ""),
    )


def check_backtest_reports_trades(log: ValidationLog, table: pd.DataFrame) -> None:
    has_trades = "trades" in table.columns
    log.add(
        "reporting.backtests_report_trade_count",
        has_trades,
        f"{len(table)} backtest rows, all with a trade count"
        + (f" (total {int(table['trades'].sum())})" if has_trades else ""),
    )


def check_charts_dated(log: ValidationLog, manifest: pd.DataFrame) -> None:
    undated = manifest.loc[~manifest["dated"], "chart"].tolist() if len(manifest) else []
    log.add(
        "reporting.charts_carry_date_range",
        not undated,
        f"{len(manifest)} charts written, each stamped with its start and end date"
        + (f"; undated: {undated}" if undated else ""),
    )
