"""Signal construction for the three hypotheses.

Every signal here returns a position series on the daily calendar where the value at
date t is the exposure decided using information available at t. The backtester applies
that exposure to the return from t to t+1, so no signal ever earns the return that
created it.

``SIGNAL_REGISTRY`` is the complete list of candidate signals evaluated by the project.
It exists so that "how many signals did you test?" has an exact answer, and so that the
multiple-testing adjustment covers the whole family rather than the survivors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import config, relative_value

# Relative-value entry and exit thresholds on the residual z-score, fixed in advance.
ENTRY_Z = 2.0
EXIT_Z = 0.5
# Positioning signal: how extreme a 3y position z-score must be to take a view.
POSITION_ENTRY_Z = 1.5
POSITION_EXIT_Z = 0.5
# Expanding-window refit cadence for the predictive equity model, in trading days.
REFIT_EVERY = 21
MIN_TRAIN_DAYS = 504


@dataclass(frozen=True)
class SignalSpec:
    """One candidate signal, declared before its performance is known."""

    name: str
    hypothesis: str
    return_column: str
    description: str
    tradable: bool


def _hysteresis(zscore: pd.Series, entry: float, exit_level: float, direction: int) -> pd.Series:
    """Threshold rule with separate entry and exit bands.

    ``direction`` is -1 for a contrarian rule (fade the extreme) and +1 for a momentum
    rule. Positions are held between the exit band and the entry band, which keeps
    turnover down relative to trading every threshold crossing.
    """
    values = zscore.to_numpy(dtype=float)
    out = np.zeros(len(values))
    state = 0.0
    for i, z in enumerate(values):
        if np.isnan(z):
            state = 0.0
        elif abs(z) < exit_level:
            state = 0.0
        elif abs(z) >= entry:
            state = direction * np.sign(z)
        out[i] = state
    return pd.Series(out, index=zscore.index)


# --------------------------------------------------------------------------------------
# Hypothesis 2: relative value
# --------------------------------------------------------------------------------------
def relative_value_positions(zscores: pd.DataFrame) -> pd.DataFrame:
    """Fade extreme residuals: short the rich leg when z is high, buy it when z is low."""
    return pd.DataFrame(
        {name: _hysteresis(zscores[name], ENTRY_Z, EXIT_Z, direction=-1) for name in relative_value.BACKTEST_PAIRS},
        index=zscores.index,
    )


# --------------------------------------------------------------------------------------
# Hypothesis 3: futures positioning
# --------------------------------------------------------------------------------------
def positioning_positions(feature_frame: pd.DataFrame) -> pd.DataFrame:
    """Fade crowded speculative positioning in each COT market.

    The signal is contrarian: a stretched long from leveraged funds or managed money is
    treated as a headwind. Whether that is right is the hypothesis, and section 12 of the
    research report reports what the data actually says.
    """
    columns = {}
    for market in config.COT_UNIVERSE:
        column = f"cot_{market.name}_z156"
        if column in feature_frame:
            columns[market.name] = _hysteresis(
                feature_frame[column], POSITION_ENTRY_Z, POSITION_EXIT_Z, direction=-1
            )
    return pd.DataFrame(columns, index=feature_frame.index)


def positioning_bucket_study(
    feature_frame: pd.DataFrame, proxy_returns: pd.DataFrame, horizons: tuple[int, ...] = (5, 10, 21)
) -> pd.DataFrame:
    """Forward returns conditioned on how crowded positioning was.

    Buckets are the quantiles requested in the brief (bottom 10%, 10-25%, 25-75%,
    75-90%, top 10%) cut on the full-sample distribution of the position z-score. That
    makes this an in-sample descriptive study, not a tradable rule: the breakpoints use
    the whole sample. The tradable version is :func:`positioning_positions`, which uses
    only trailing information.

    Forward windows overlap heavily, so the reported t-statistics use Newey-West
    standard errors with a bandwidth of the horizon length. Naive standard errors would
    be far too small.
    """
    edges = [0.0, 0.10, 0.25, 0.75, 0.90, 1.0]
    labels = ["bottom 10%", "10-25%", "25-75%", "75-90%", "top 10%"]
    rows = []

    for market in config.COT_UNIVERSE:
        column = f"cot_{market.name}_z156"
        if column not in feature_frame:
            continue
        zscore = feature_frame[column].dropna()
        returns = proxy_returns[market.price_proxy]
        buckets = pd.cut(
            zscore.rank(pct=True), bins=edges, labels=labels, include_lowest=True, ordered=True
        )

        for horizon in horizons:
            # Forward sum of log returns over the next `horizon` sessions.
            forward = returns.shift(-1).rolling(horizon).sum().shift(-(horizon - 1))
            forward_vol = returns.shift(-1).rolling(horizon).std().shift(-(horizon - 1)) * np.sqrt(
                config.TRADING_DAYS_PER_YEAR
            )
            cumulative = returns.shift(-1).rolling(horizon).apply(
                lambda window: float((np.cumsum(window) - np.maximum.accumulate(np.cumsum(window))).min()), raw=True
            ).shift(-(horizon - 1))

            data = pd.concat(
                {"bucket": buckets, "forward": forward, "vol": forward_vol, "drawdown": cumulative},
                axis=1,
                sort=True,
            ).dropna()
            for label in labels:
                subset = data[data["bucket"] == label]
                if len(subset) < 30:
                    continue
                model = sm.OLS(subset["forward"].to_numpy(), np.ones(len(subset))).fit(
                    cov_type="HAC", cov_kwds={"maxlags": horizon}
                )
                rows.append(
                    {
                        "market": market.name,
                        "label": market.label,
                        "horizon_days": horizon,
                        "bucket": label,
                        "nobs": int(len(subset)),
                        "mean_return_pct": float(subset["forward"].mean()) * 100.0,
                        "t_stat_hac": float(model.tvalues[0]),
                        "p_value_hac": float(model.pvalues[0]),
                        "hit_rate": float((subset["forward"] > 0).mean()),
                        "forward_vol": float(subset["vol"].mean()),
                        "mean_drawdown_pct": float(subset["drawdown"].mean()) * 100.0,
                    }
                )
    return pd.DataFrame(rows)


def positioning_extremes_test(bucket_study: pd.DataFrame) -> pd.DataFrame:
    """Top decile minus bottom decile forward return, per market and horizon."""
    rows = []
    for (market, horizon), group in bucket_study.groupby(["market", "horizon_days"]):
        indexed = group.set_index("bucket")
        if "top 10%" not in indexed.index or "bottom 10%" not in indexed.index:
            continue
        top, bottom = indexed.loc["top 10%"], indexed.loc["bottom 10%"]
        # Standard errors implied by each bucket's HAC t-statistic.
        se_top = abs(top["mean_return_pct"] / top["t_stat_hac"]) if top["t_stat_hac"] else np.nan
        se_bottom = abs(bottom["mean_return_pct"] / bottom["t_stat_hac"]) if bottom["t_stat_hac"] else np.nan
        difference = top["mean_return_pct"] - bottom["mean_return_pct"]
        se = float(np.sqrt(se_top**2 + se_bottom**2))
        rows.append(
            {
                "market": market,
                "horizon_days": int(horizon),
                "top_decile_pct": top["mean_return_pct"],
                "bottom_decile_pct": bottom["mean_return_pct"],
                "difference_pct": difference,
                "std_error": se,
                "t_stat": difference / se if se else np.nan,
                "nobs_top": int(top["nobs"]),
                "nobs_bottom": int(bottom["nobs"]),
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["significant_5pct"] = frame["t_stat"].abs() > 1.96
    return frame


# --------------------------------------------------------------------------------------
# Hypothesis 1: the predictive version of the equity regression
# --------------------------------------------------------------------------------------
def equity_prediction_positions(feature_frame: pd.DataFrame, regressors: list[str]) -> pd.Series:
    """Walk-forward one-day-ahead equity forecast, traded as a long/short position.

    The model is refit on an expanding window every ``REFIT_EVERY`` sessions using only
    data strictly before the forecast date, and the position is the sign of the forecast.
    Hypothesis 1's contemporaneous regression is not tradable; this is the version that
    is, and its performance is the honest test of whether the relationship has any
    forecasting content.
    """
    target = "ret_1d_spx"
    frame = feature_frame[[target, *regressors]].copy()
    frame[regressors] = frame[regressors].shift(1)
    frame = frame.dropna()

    positions = pd.Series(0.0, index=frame.index)
    model = None
    for i in range(MIN_TRAIN_DAYS, len(frame)):
        if model is None or (i - MIN_TRAIN_DAYS) % REFIT_EVERY == 0:
            history = frame.iloc[:i]
            model = sm.OLS(history[target], sm.add_constant(history[regressors])).fit()
        row = frame.iloc[[i]]
        forecast = float(model.predict(sm.add_constant(row[regressors], has_constant="add")).iloc[0])
        positions.iloc[i] = float(np.sign(forecast))
    return positions.reindex(feature_frame.index).fillna(0.0)


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------
def build_registry() -> list[SignalSpec]:
    """Every candidate signal the project evaluates.

    The relative-value pairs appear twice where both constructions are implementable:
    once as the beta-hedged spread and once as an outright position in the tradable leg.
    Both are reported, because the difference between them is one of the project's
    findings rather than a detail.
    """
    specs = [
        SignalSpec(
            "h1_equity_forecast",
            "H1 rates and equity risk",
            "ret_1d_spx",
            "Sign of a walk-forward one-day-ahead S&P 500 forecast from lagged real "
            "yield, breakeven, curve slope and VIX changes.",
            tradable=True,
        )
    ]
    for pair in relative_value.PAIRS:
        rule = f"Fade the rolling-beta residual of {pair.y} on {pair.x} at |z| > {ENTRY_Z}, exit at |z| < {EXIT_Z}."
        if pair.return_scale is not None:
            specs.append(
                SignalSpec(
                    f"h2_rv_{pair.name}_hedged",
                    "H2 cross-asset relative value",
                    f"hedged_{pair.name}",
                    f"{rule} Traded as the beta-hedged spread.",
                    tradable=True,
                )
            )
        if pair.name in relative_value.SINGLE_LEG_PAIRS:
            specs.append(
                SignalSpec(
                    f"h2_rv_{pair.name}_single",
                    "H2 cross-asset relative value",
                    f"ret_1d_{pair.tradable_leg}",
                    f"{rule} Traded as an outright position in {pair.tradable_leg} only.",
                    tradable=True,
                )
            )
        if pair.return_scale is None:
            specs.append(
                SignalSpec(
                    f"h2_rv_{pair.name}",
                    "H2 cross-asset relative value",
                    pair.tradable_leg,
                    f"{rule} Analysed only: {pair.tradable_note}",
                    tradable=False,
                )
            )
    for market in config.COT_UNIVERSE:
        specs.append(
            SignalSpec(
                f"h3_positioning_{market.name}",
                "H3 futures positioning",
                market.price_proxy,
                f"Fade {config.COT_SPEC_CATEGORY[market.report].replace('_', ' ')} crowding in "
                f"{market.label} at |3y z| > {POSITION_ENTRY_Z}.",
                tradable=True,
            )
        )
    return specs


SIGNAL_REGISTRY = build_registry()


def registry_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "signal": s.name,
                "hypothesis": s.hypothesis,
                "return_series": s.return_column,
                "backtested": s.tradable,
                "description": s.description,
            }
            for s in SIGNAL_REGISTRY
        ]
    )
