"""Feature engine.

Three rules hold everywhere in this module:

1. Price series become log returns; yield and spread series become basis point changes.
   Raw levels are never fed to a regression.
2. Every window is trailing and closed at the observation date. No centred windows, no
   full-sample moments, no interpolation across a gap.
3. Weekly and monthly inputs are stamped with the date they became public, then
   forward-filled onto the daily calendar from that date. A value is never visible on a
   date before it existed.

``validation.check_causality`` enforces rule 2 empirically against
:func:`build_feature_frame`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, trading_calendar


# --------------------------------------------------------------------------------------
# Transforms
# --------------------------------------------------------------------------------------
def log_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """r_t = ln(P_t / P_{t-1})."""
    return np.log(prices).diff()


def bp_changes(yields: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """dy_t in basis points, from yields stored in percent."""
    return yields.diff() * config.BP_PER_PCT


def rolling_zscore(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """z_t = (x_t - mean of the trailing window) / sd of the trailing window."""
    min_periods = min_periods or max(20, window // 2)
    mean = series.rolling(window, min_periods=min_periods).mean()
    sd = series.rolling(window, min_periods=min_periods).std()
    return (series - mean) / sd.replace(0.0, np.nan)


def rolling_percentile(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Percentile rank of the current value within its own trailing window."""
    min_periods = min_periods or max(20, window // 2)
    return series.rolling(window, min_periods=min_periods).rank(pct=True) * 100.0


def annualised_vol(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window, min_periods=max(10, window // 2)).std() * np.sqrt(config.TRADING_DAYS_PER_YEAR)


def curves(panel: pd.DataFrame) -> pd.DataFrame:
    """Yield curve spreads in basis points."""
    out = pd.DataFrame(index=panel.index)
    for name, (long_leg, short_leg) in config.CURVE_DEFS.items():
        out[name] = (panel[long_leg] - panel[short_leg]) * config.BP_PER_PCT
    return out


def return_proxies(panel: pd.DataFrame) -> pd.DataFrame:
    """Tradable-return proxies for the CFTC positioning markets.

    These are approximations, not futures returns; each one is documented in
    ``config.COT_UNIVERSE[*].proxy_note``. The 10y note proxy is -D x dy with the
    modified duration in ``config.ZN_MODIFIED_DURATION``; the euro proxy is the negated
    broad-dollar return; the rest are cash log returns.
    """
    out = pd.DataFrame(index=panel.index)
    out["zn_proxy"] = -config.ZN_MODIFIED_DURATION * panel["dgs10"].diff() / config.BP_PER_PCT
    out["usd_short_proxy"] = -log_returns(panel["usd"])
    return out


# --------------------------------------------------------------------------------------
# Publication-aware alignment
# --------------------------------------------------------------------------------------
def cot_available_from(report_dates: pd.Series) -> pd.Series:
    """First trading session on which a COT report may be used.

    Positions are measured at Tuesday's close and published the following Friday at
    15:30 US/Eastern. 15:30 ET is inside the cash session, so rather than assume a fill
    in the last half hour the platform waits for the next trading session. From a
    Tuesday as-of date that is the following Monday: a six calendar day lag.
    """
    sessions = trading_calendar.trading_days(config.START_DATE, config.END_DATE + pd.Timedelta(days=10))
    release = report_dates + pd.Timedelta(days=config.COT_RELEASE_WEEKDAY - config.COT_AS_OF_WEEKDAY)
    positions = sessions.searchsorted(release.to_numpy(), side="right")
    positions = np.clip(positions, 0, len(sessions) - 1)
    return pd.Series(sessions[positions], index=report_dates.index, name="available_from")


def cot_features(cot: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Daily positioning features, visible only from their publication session.

    For each market's speculative category (leveraged funds for financial contracts,
    managed money for physical commodities):

        net position   = long - short
        position ratio = (long - short) / open interest
        z score        = 3y (156 week) trailing z score of the position ratio
        percentile     = 52 week trailing percentile of the position ratio
        weekly change  = week-on-week change in the position ratio
    """
    frames = []
    for market in config.COT_UNIVERSE:
        category = config.COT_SPEC_CATEGORY[market.report]
        weekly = (
            cot[(cot["market"] == market.name) & (cot["category"] == category)]
            .sort_values("report_date")
            .reset_index(drop=True)
        )
        weekly["available_from"] = cot_available_from(weekly["report_date"])
        net = weekly["long"] - weekly["short"]
        ratio = net / weekly["open_interest"]

        block = pd.DataFrame(
            {
                f"cot_{market.name}_net": net,
                f"cot_{market.name}_ratio": ratio,
                f"cot_{market.name}_z156": rolling_zscore(ratio, config.COT_ZSCORE_WEEKS, min_periods=52),
                f"cot_{market.name}_pct52w": rolling_percentile(ratio, 52, min_periods=26),
                f"cot_{market.name}_chg1w": ratio.diff(),
                f"cot_{market.name}_oi": weekly["open_interest"],
            }
        )
        block.index = pd.DatetimeIndex(weekly["available_from"])
        # Two reports can land on one session only if a release is skipped; keep the last.
        block = block[~block.index.duplicated(keep="last")]
        frames.append(block.reindex(index.union(block.index)).ffill().reindex(index))

    out = pd.concat(frames, axis=1)
    out.index.name = "date"
    return out


def macro_point_in_time(vintages: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Point-in-time macro features built from ALFRED-style vintages.

    At each vintage date the series is reconstructed *as it was known then* (for every
    reference period, the value from the latest vintage at or before that date), the
    transform is computed on that reconstruction, and the result is forward-filled onto
    the daily calendar. A 2019 backtest therefore sees the 2019 print, not today's
    revised value.
    """
    transforms = {
        "cpi": lambda s: (s / s.shift(12) - 1.0) * 100.0,
        "payrolls": lambda s: s.diff().rolling(3).mean(),
        "unrate": lambda s: s,
        "indpro": lambda s: (s / s.shift(12) - 1.0) * 100.0,
        "gdp": lambda s: (s / s.shift(4) - 1.0) * 100.0,
    }
    columns = {}
    for name, transform in transforms.items():
        rows = vintages[vintages["series"] == name].sort_values(["vintage_date", "reference_period"])
        if rows.empty:
            continue
        known: dict[pd.Timestamp, float] = {}
        observations = []
        for vintage_date, group in rows.groupby("vintage_date", sort=True):
            for period, value in zip(group["reference_period"], group["value"]):
                known[period] = value
            as_of = pd.Series(known).sort_index()
            transformed = transform(as_of)
            latest = transformed.dropna()
            if not latest.empty:
                observations.append((vintage_date, float(latest.iloc[-1])))
        if observations:
            dates, values = zip(*observations)
            series = pd.Series(values, index=pd.DatetimeIndex(dates))
            series = series[~series.index.duplicated(keep="last")]
            columns[f"macro_{name}"] = series.reindex(index.union(series.index)).ffill().reindex(index)

    out = pd.DataFrame(columns, index=index)
    out.index.name = "date"
    return out


# --------------------------------------------------------------------------------------
# Feature frame
# --------------------------------------------------------------------------------------
def build_feature_frame(panel: pd.DataFrame) -> pd.DataFrame:
    """Daily market features: returns, changes, curves, volatility and z-scores.

    COT and macro features are added separately by :func:`build_full_features` because
    they need their own source frames; keeping them out of here lets the causality check
    perturb the daily panel in isolation.
    """
    price_columns = [c for c in config.PRICE_COLUMNS if c in panel.columns]
    yield_columns = [c for c in config.YIELD_COLUMNS if c in panel.columns]
    curve_frame = curves(panel)
    columns: dict[str, pd.Series] = {}

    for column in price_columns:
        columns[f"ret_1d_{column}"] = log_returns(panel[column])
    for column in yield_columns:
        columns[f"dy_1d_{column}"] = bp_changes(panel[column])
    for column in curve_frame.columns:
        columns[column] = curve_frame[column]
        columns[f"d_1d_{column}"] = curve_frame[column].diff()

    # Level and slope decomposition of the nominal curve. Delta level and delta slope
    # span the same space as (dy2, dy10) without the exact collinearity that including
    # dy2, dy10 and dcurve together would create.
    columns["dy_1d_level"] = 0.5 * (columns["dy_1d_dgs2"] + columns["dy_1d_dgs10"])
    columns["dy_1d_slope"] = columns["dy_1d_dgs10"] - columns["dy_1d_dgs2"]

    for horizon in config.HORIZONS:
        if horizon == 1:
            continue
        for column in price_columns:
            columns[f"ret_{horizon}d_{column}"] = np.log(panel[column]).diff(horizon)
        for column in yield_columns:
            columns[f"dy_{horizon}d_{column}"] = panel[column].diff(horizon) * config.BP_PER_PCT
        for column in curve_frame.columns:
            columns[f"d_{horizon}d_{column}"] = curve_frame[column].diff(horizon)

    for window in config.VOL_WINDOWS:
        for column in price_columns:
            columns[f"vol_{window}_{column}"] = annualised_vol(columns[f"ret_1d_{column}"], window)

    levels = pd.concat([panel, curve_frame], axis=1)
    for window in config.ZSCORE_WINDOWS:
        for column in levels.columns:
            columns[f"z{window}_{column}"] = rolling_zscore(levels[column], window)

    out = pd.concat({**columns, **dict(return_proxies(panel).items())}, axis=1)
    out.index.name = "date"
    return out


def build_full_features(
    panel: pd.DataFrame, cot: pd.DataFrame, macro_vintages: pd.DataFrame
) -> pd.DataFrame:
    """Market, positioning and macro features on one daily index."""
    market = build_feature_frame(panel)
    positioning = cot_features(cot, panel.index)
    macro = macro_point_in_time(macro_vintages, panel.index)
    return pd.concat([market, positioning, macro], axis=1)
