"""Frozen synthetic dataset for the research window.

WHY THIS EXISTS
---------------
The platform's primary data path is live FRED / ALFRED / CFTC / Cboe ingestion
(``src/data_loader.py``). This module is the offline fallback: it generates a fixed,
seeded dataset with the same schema, calendar and units so that the feature engine,
regressions, stationarity tests, backtester, Excel monitor and test suite can all be
executed and verified without network access.

WHAT IS REAL AND WHAT IS NOT
----------------------------
Only three numbers in this module are observed market data, and they are the project
anchors declared in ``config.ANCHORS``: the 2026-08-13 S&P 500 close (7798.99), the
2y Treasury yield (4.15%) and the 10y Treasury yield (4.63%). The generator is pinned
to reproduce those three exactly so that ``validation.check_anchor_values`` exercises a
real code path. Every other level, path and correlation here is simulated from the
documented scenario below and must not be quoted as market history.

Any result computed from this dataset is a property of this simulator, not of markets.

SCENARIO DESIGN
---------------
Trends are piecewise-linear through documented knots that reproduce the *shape* of a
policy cycle (near-zero rates, a hiking cycle that inverts the curve, then partial
cuts) so that the regime-classification and rolling-beta machinery has curve
inversions, vol spikes and sign changes to find. Around each trend, innovations are
built from a small set of shared shocks so that cross-asset correlations arise from a
factor structure rather than being imposed pairwise:

    z_risk    global risk appetite      z_dollar  broad dollar
    z_real    real interest rates       z_oil     oil supply
    z_infl    inflation expectations    z_gold    gold-specific flows

Equity loads on ``z_real`` with a loading that changes sign mid-sample, mirroring the
2016-2020 "good news is good news" regime giving way to the 2022+ inflation regime.
Gold carries an Ornstein-Uhlenbeck flow wedge, so a gold / real-yield residual has
genuine mean reversion for the relative-value layer to measure.

Noise is applied as a Brownian bridge, so every series terminates exactly on its
scenario endpoint and the anchor pin is exact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, trading_calendar

# --------------------------------------------------------------------------------------
# Scenario knots. Values marked REAL are observed data; all others are simulated.
# --------------------------------------------------------------------------------------
SCENARIO_NOTE = (
    "Trend knots approximate published historical levels so the monitor and regime "
    "layer operate on a plausibly shaped cycle. All daily variation, cross-asset "
    "correlations, positioning data, macro vintages and every derived statistic are "
    "simulated. Levels from 2025-08 onward are hypothetical apart from the three "
    "config.ANCHORS observations."
)

YIELD_KNOTS: dict[str, list[tuple[str, float]]] = {
    "dgs2": [
        ("2016-08-15", 0.72),
        ("2016-12-30", 1.20),
        ("2017-12-29", 1.89),
        ("2018-11-08", 2.97),
        ("2019-12-31", 1.58),
        ("2020-03-31", 0.25),
        ("2020-12-31", 0.13),
        ("2021-12-31", 0.73),
        ("2022-12-30", 4.41),
        ("2023-07-31", 4.88),
        ("2023-10-31", 5.05),
        ("2024-12-31", 4.25),
        ("2025-12-31", 4.05),
        ("2026-08-13", 4.15),  # REAL anchor
    ],
    "real10": [
        ("2016-08-15", 0.13),
        ("2016-12-30", 0.47),
        ("2017-12-29", 0.44),
        ("2018-11-08", 1.15),
        ("2019-12-31", 0.15),
        ("2020-03-31", -0.17),
        ("2020-08-31", -1.08),
        ("2020-12-31", -1.06),
        ("2021-12-31", -1.04),
        ("2022-12-30", 1.58),
        ("2023-07-31", 1.55),
        ("2023-10-31", 2.45),
        ("2024-12-31", 2.24),
        ("2026-08-13", 2.20),
    ],
    "breakeven10": [
        ("2016-08-15", 1.43),
        ("2016-12-30", 1.97),
        ("2017-12-29", 1.98),
        ("2018-11-08", 2.05),
        ("2019-12-31", 1.77),
        ("2020-03-31", 0.95),
        ("2020-12-31", 1.99),
        ("2021-12-31", 2.56),
        ("2022-04-29", 3.05),
        ("2022-12-30", 2.30),
        ("2023-07-31", 2.32),
        ("2023-10-31", 2.42),
        ("2024-12-31", 2.34),
        ("2026-08-13", 2.43),  # = 4.63 (REAL anchor) - 2.20 (simulated real yield)
    ],
    "term_spread_30_10": [  # 30y minus 10y
        ("2016-08-15", 0.72),
        ("2017-12-29", 0.33),
        ("2019-08-28", 0.42),
        ("2020-12-31", 0.77),
        ("2021-03-31", 0.85),
        ("2022-12-30", 0.09),
        ("2023-10-31", 0.15),
        ("2024-12-31", 0.20),
        ("2026-08-13", 0.62),
    ],
    "fedfunds": [
        ("2016-08-15", 0.40),
        ("2016-12-31", 0.55),
        ("2017-12-31", 1.33),
        ("2018-12-31", 2.40),
        ("2019-12-31", 1.55),
        ("2020-04-30", 0.05),
        ("2022-03-31", 0.33),
        ("2022-12-31", 4.33),
        ("2023-08-31", 5.33),
        ("2024-09-30", 5.33),
        ("2024-12-31", 4.33),
        ("2025-12-31", 3.83),
        ("2026-08-13", 3.58),
    ],
    "hy_oas": [
        ("2016-08-15", 5.10),
        ("2016-12-30", 4.22),
        ("2017-12-29", 3.63),
        ("2018-12-31", 5.33),
        ("2019-12-31", 3.60),
        ("2020-03-23", 10.87),
        ("2020-12-31", 3.60),
        ("2021-12-31", 3.10),
        ("2022-07-05", 5.99),
        ("2022-12-30", 4.81),
        ("2023-12-29", 3.39),
        ("2024-12-31", 2.92),
        ("2026-08-13", 3.15),
    ],
}

PRICE_KNOTS: dict[str, list[tuple[str, float]]] = {
    "spx": [
        ("2016-08-15", 2190.15),
        ("2016-12-30", 2238.83),
        ("2017-12-29", 2673.61),
        ("2018-12-24", 2351.10),
        ("2019-12-31", 3230.78),
        ("2020-03-23", 2237.40),
        ("2020-12-31", 3756.07),
        ("2021-12-31", 4766.18),
        ("2022-10-12", 3577.03),
        ("2022-12-30", 3839.50),
        ("2023-12-29", 4769.83),
        ("2024-12-31", 5881.63),
        ("2025-12-31", 6900.00),
        ("2026-08-13", 7798.99),  # REAL anchor
    ],
    "ndx": [
        ("2016-08-15", 4800.00),
        ("2016-12-30", 4863.62),
        ("2017-12-29", 6396.42),
        ("2018-12-24", 5715.00),
        ("2019-12-31", 8733.07),
        ("2020-03-23", 6994.29),
        ("2020-12-31", 12888.28),
        ("2021-12-31", 16320.08),
        ("2022-12-30", 10939.76),
        ("2023-12-29", 16825.93),
        ("2024-12-31", 21012.17),
        ("2025-12-31", 25200.00),
        ("2026-08-13", 28450.00),
    ],
    "usd": [
        ("2016-08-15", 119.00),
        ("2016-12-30", 122.50),
        ("2017-12-29", 113.50),
        ("2018-12-31", 118.00),
        ("2019-12-31", 118.50),
        ("2020-03-23", 126.50),
        ("2020-12-31", 112.00),
        ("2021-12-31", 115.50),
        ("2022-09-27", 128.60),
        ("2022-12-30", 121.50),
        ("2023-12-29", 118.00),
        ("2024-12-31", 122.10),
        ("2026-08-13", 116.50),
    ],
    "wti": [
        ("2016-08-15", 45.70),
        ("2016-12-30", 53.72),
        ("2017-12-29", 60.42),
        ("2018-10-03", 76.40),
        ("2018-12-31", 45.41),
        ("2019-12-31", 61.14),
        # FRED's DCOILWTICO printed -37.63 on 2020-04-20. The scenario floors the
        # trough at the lowest positive print; the live loader must handle the
        # negative observation explicitly (see validation.check_positive_prices).
        ("2020-04-21", 11.57),
        ("2020-12-31", 48.52),
        ("2021-12-31", 75.21),
        ("2022-06-08", 122.11),
        ("2022-12-30", 80.26),
        ("2023-12-29", 71.65),
        ("2024-12-31", 71.20),
        ("2026-08-13", 64.50),
    ],
    "gold": [
        ("2016-08-15", 1345.00),
        ("2016-12-30", 1145.90),
        ("2017-12-29", 1291.00),
        ("2018-09-28", 1187.25),
        ("2019-12-31", 1515.00),
        ("2020-08-06", 2063.55),
        ("2020-12-31", 1887.60),
        ("2021-12-31", 1805.85),
        ("2022-09-27", 1622.30),
        ("2023-12-29", 2062.40),
        ("2024-12-31", 2610.00),
        ("2026-08-13", 3860.00),
    ],
}

VIX_KNOTS = [
    ("2016-08-15", 12.30),
    ("2017-12-29", 11.04),
    ("2018-02-05", 37.32),
    ("2018-12-24", 36.07),
    ("2019-12-31", 13.78),
    ("2020-03-16", 82.69),
    ("2020-12-31", 22.75),
    ("2021-12-31", 17.22),
    ("2022-10-11", 33.63),
    ("2023-12-29", 12.45),
    ("2024-08-05", 38.57),
    ("2024-12-31", 17.35),
    ("2026-08-13", 16.40),
]

# Plausibility floors. US nominal Treasury yields did not go negative in this window,
# so simulated noise is prevented from pushing levels outside a credible range.
LEVEL_FLOORS = {"dgs2": 0.03, "dgs10": 0.03, "dgs30": 0.30, "real10": -2.10, "breakeven10": 0.25, "hy_oas": 2.20}

# Yield deviations from trend: factor loadings plus the mean-reversion calibration.
# ``sd_bp`` is the stationary sd of the deviation; ``halflife`` is in trading days.
YIELD_OU: dict[str, dict] = {
    "real10": {
        "loadings": {"z_real": 4.2, "z_risk": -1.1},
        "idio_sd": 1.9,
        "halflife": 70.0,
        "sd_bp": 34.0,
    },
    "breakeven10": {
        "loadings": {"z_infl": 2.1, "z_oil": 1.3, "z_risk": 0.9},
        "idio_sd": 1.4,
        "halflife": 55.0,
        "sd_bp": 22.0,
    },
    "dgs2": {
        "loadings": {"z_real": 3.4, "z_risk": -0.8, "z_infl": 1.0},
        "idio_sd": 1.7,
        "halflife": 65.0,
        "sd_bp": 30.0,
    },
    "term_spread_30_10": {
        "loadings": {"z_real": -0.9, "z_infl": 0.5},
        "idio_sd": 0.9,
        "halflife": 50.0,
        "sd_bp": 12.0,
    },
    "hy_oas": {
        "loadings": {"z_risk": -2.6},
        "idio_sd": 2.2,
        "halflife": 45.0,
        "sd_bp": 48.0,
    },
}

# Price loadings, in log return per unit shock.
LOADINGS = {
    "spx_ret": {"z_risk": 0.0082},
    "usd_ret": {"z_dollar": 0.00231, "z_real": 0.00098, "z_risk": -0.00080},
    "wti_ret": {"z_oil": 0.0172, "z_risk": 0.0068, "z_dollar": -0.0039},
    "gold_ret": {"z_real": -0.0040, "z_dollar": -0.0031, "z_infl": 0.0014, "z_gold": 0.0032},
}
# VIX deviation from its scenario trend: loading on the stochastic volatility factor
# plus a fast mean-reverting component that carries the daily equity/VIX correlation.
VIX_VOL_LOADING = 0.85
VIX_FAST_LOADINGS = {"z_risk": -0.80, "z_vol": 0.60}
VIX_FAST_HALFLIFE = 12.0
VIX_FAST_SD = 0.10
# Equity loading on the real-rate shock, transitioning from positive to negative.
SPX_REAL_LOADING_EARLY = 0.0021
SPX_REAL_LOADING_LATE = -0.0034
SPX_REGIME_MIDPOINT = pd.Timestamp("2021-09-30")
SPX_REGIME_WIDTH_DAYS = 420.0
NDX_BETA_TO_SPX = 1.16

# Ornstein-Uhlenbeck flow wedge in log gold: amplitude and mean-reversion half-life.
# Kept deliberately modest so the relative-value residual is tradable but not trivial.
GOLD_WEDGE_SD = 0.018
GOLD_WEDGE_HALFLIFE_DAYS = 30.0

# Stochastic volatility factor: OU in logs with seeded jumps.
VOL_HALFLIFE_DAYS = 34.0
VOL_SD = 0.30
VOL_JUMP_PROB = 0.0022
VOL_JUMP_SIZE = 0.75
VOL_RISK_LOADING = 0.55

COT_BASE_OPEN_INTEREST = {"es": 2_450_000, "zn": 4_100_000, "6e": 690_000, "cl": 1_900_000, "gc": 520_000}
# Gross long+short exposure per category as a share of open interest. Each report's
# shares sum to 1.76, leaving 12% of open interest to non-reportable traders so that
# reportable + non-reportable longs reconcile to open interest, as in real COT data.
COT_GROSS_SHARE = {
    "tff": {"dealer": 0.52, "asset_manager": 0.46, "leveraged_funds": 0.42, "other_rept": 0.36},
    "disagg": {"prod_merc": 0.62, "swap_dealer": 0.44, "managed_money": 0.38, "other_rept": 0.32},
}


def _piecewise(knots: list[tuple[str, float]], index: pd.DatetimeIndex) -> pd.Series:
    """Linear interpolation through dated knots, evaluated on ``index``."""
    dates = pd.DatetimeIndex([pd.Timestamp(d) for d, _ in knots])
    values = np.array([v for _, v in knots], dtype=float)
    x = index.astype("int64").to_numpy(dtype=float)
    xp = dates.astype("int64").to_numpy(dtype=float)
    return pd.Series(np.interp(x, xp, values), index=index)


def _pin(path: np.ndarray) -> np.ndarray:
    """Tilt a path by a linear ramp so that it terminates at exactly zero."""
    n = len(path)
    return path - path[-1] * (np.arange(1, n + 1) / n)


def _bridge(noise: np.ndarray) -> np.ndarray:
    """Cumulative sum of ``noise`` pinned to zero at both ends (Brownian bridge)."""
    return _pin(np.cumsum(noise))


def _ou(shocks: np.ndarray, halflife: float, sd: float) -> np.ndarray:
    """Ornstein-Uhlenbeck path with the given half-life and stationary sd.

    ``shocks`` is standardised first so callers can pass an arbitrary linear
    combination of factor shocks without changing the target volatility.
    """
    z = np.asarray(shocks, dtype=float)
    z = z / z.std()
    phi = 0.5 ** (1.0 / halflife)
    innovation_sd = sd * np.sqrt(1.0 - phi**2)
    out = np.empty(len(z))
    level = 0.0
    for i, s in enumerate(z):
        level = phi * level + innovation_sd * s
        out[i] = level
    return out


def _vol_factor(rng: np.random.Generator, n: int, risk_shock: np.ndarray) -> np.ndarray:
    """Mean-reverting log-volatility multiplier with occasional upward jumps.

    The driving shock loads negatively on risk appetite, so volatility rises when
    equities fall. That is what gives the panel a realistic equity/VIX correlation and
    negative return skew.
    """
    phi = 0.5 ** (1.0 / VOL_HALFLIFE_DAYS)
    innovation_sd = VOL_SD * np.sqrt(1.0 - phi**2)
    driver = VOL_RISK_LOADING * -risk_shock + np.sqrt(1.0 - VOL_RISK_LOADING**2) * rng.standard_normal(n)
    shocks = driver / driver.std()
    jumps = (rng.random(n) < VOL_JUMP_PROB) * VOL_JUMP_SIZE
    log_v = np.empty(n)
    level = 0.0
    for i in range(n):
        level = phi * level + innovation_sd * shocks[i] + jumps[i]
        log_v[i] = level
    return np.exp(log_v - log_v.mean())


def _spx_real_loading(index: pd.DatetimeIndex) -> np.ndarray:
    """Smooth transition of the equity/real-rate loading from positive to negative."""
    days = (index - SPX_REGIME_MIDPOINT).days.to_numpy(dtype=float)
    weight = 1.0 / (1.0 + np.exp(-days / (SPX_REGIME_WIDTH_DAYS / 4.0)))
    return SPX_REAL_LOADING_EARLY + weight * (SPX_REAL_LOADING_LATE - SPX_REAL_LOADING_EARLY)


def _composite(loadings: dict[str, float], shocks: dict[str, np.ndarray], idio: np.ndarray, idio_sd: float) -> np.ndarray:
    out = idio_sd * idio
    for key, beta in loadings.items():
        out = out + beta * shocks[key]
    return out


def _yield_path(
    name: str,
    index: pd.DatetimeIndex,
    shocks: dict[str, np.ndarray],
    idio: np.ndarray,
    vol: np.ndarray,
) -> pd.Series:
    """Scenario trend plus a pinned mean-reverting deviation, in percent.

    Yields get an Ornstein-Uhlenbeck deviation rather than a random walk: over 2,500
    days a random walk with realistic daily volatility drifts hundreds of basis points
    away from any trend and produces implausible levels.
    """
    spec = YIELD_OU[name]
    composite = _composite(spec["loadings"], shocks, idio, spec["idio_sd"])
    deviation = _ou(composite * vol, spec["halflife"], spec["sd_bp"])
    return _piecewise(YIELD_KNOTS[name], index) + _pin(deviation) / config.BP_PER_PCT


def build_daily_panel(seed: int = config.SYNTHETIC_SEED) -> pd.DataFrame:
    """Daily cross-asset panel on the NYSE calendar for the frozen window."""
    index = trading_calendar.trading_days()
    n = len(index)
    rng = np.random.default_rng(seed)

    shocks = {k: rng.standard_normal(n) for k in ("z_risk", "z_real", "z_infl", "z_dollar", "z_oil", "z_gold", "z_vol")}
    vol = _vol_factor(rng, n, shocks["z_risk"])

    out = pd.DataFrame(index=index)

    # ---- Rates -----------------------------------------------------------------------
    # Floors are applied to the independent legs first, then the curve identities are
    # derived, so dgs10 = real10 + breakeven10 and dgs30 = dgs10 + term spread always hold.
    out["real10"] = _yield_path("real10", index, shocks, rng.standard_normal(n), vol).clip(lower=LEVEL_FLOORS["real10"])
    out["breakeven10"] = _yield_path("breakeven10", index, shocks, rng.standard_normal(n), vol).clip(
        lower=LEVEL_FLOORS["breakeven10"]
    )
    out["dgs2"] = _yield_path("dgs2", index, shocks, rng.standard_normal(n), vol).clip(lower=LEVEL_FLOORS["dgs2"])
    out["hy_oas"] = _yield_path("hy_oas", index, shocks, rng.standard_normal(n), vol).clip(
        lower=LEVEL_FLOORS["hy_oas"]
    )
    out["dgs10"] = out["real10"] + out["breakeven10"]
    term_spread = _yield_path("term_spread_30_10", index, shocks, rng.standard_normal(n), vol)
    out["dgs30"] = out["dgs10"] + term_spread

    # Policy rate moves in 25bp steps and is otherwise flat.
    out["fedfunds"] = (_piecewise(YIELD_KNOTS["fedfunds"], index) * 4.0).round() / 4.0

    # ---- Prices ----------------------------------------------------------------------
    spx_innovation = (
        LOADINGS["spx_ret"]["z_risk"] * shocks["z_risk"]
        + _spx_real_loading(index) * shocks["z_real"]
        + 0.0018 * rng.standard_normal(n)
    ) * vol
    out["spx"] = np.exp(np.log(_piecewise(PRICE_KNOTS["spx"], index).to_numpy()) + _bridge(spx_innovation))

    ndx_innovation = NDX_BETA_TO_SPX * spx_innovation + 0.0038 * rng.standard_normal(n) * vol
    out["ndx"] = np.exp(np.log(_piecewise(PRICE_KNOTS["ndx"], index).to_numpy()) + _bridge(ndx_innovation))

    for name, loading_key, idio_sd in (("usd", "usd_ret", 0.0011), ("wti", "wti_ret", 0.0068)):
        innovation = _composite(LOADINGS[loading_key], shocks, rng.standard_normal(n), idio_sd) * vol
        trend = np.log(_piecewise(PRICE_KNOTS[name], index).to_numpy())
        out[name] = np.exp(trend + _bridge(innovation))

    gold_innovation = _composite(LOADINGS["gold_ret"], shocks, rng.standard_normal(n), 0.0022) * vol
    gold_wedge = _ou(rng.standard_normal(n), GOLD_WEDGE_HALFLIFE_DAYS, GOLD_WEDGE_SD)
    log_gold = (
        np.log(_piecewise(PRICE_KNOTS["gold"], index).to_numpy())
        + _bridge(gold_innovation)
        + _pin(gold_wedge)
    )
    out["gold"] = np.exp(log_gold)

    # ---- Volatility ------------------------------------------------------------------
    vix_fast = _ou(_composite(VIX_FAST_LOADINGS, shocks, rng.standard_normal(n), 0.15), VIX_FAST_HALFLIFE, VIX_FAST_SD)
    vix_deviation = VIX_VOL_LOADING * np.log(vol) + vix_fast
    log_vix = np.log(_piecewise(VIX_KNOTS, index).to_numpy()) + _pin(vix_deviation)
    out["vix"] = pd.Series(np.exp(log_vix), index=index).clip(lower=8.5)

    out = out[list(config.DAILY_COLUMNS)]
    out.index.name = "date"
    return out


def build_macro_vintages(seed: int = config.SYNTHETIC_SEED + 1) -> pd.DataFrame:
    """ALFRED-shaped vintage table: one row per (series, reference period, vintage).

    The first print of each reference period differs from the final value, and later
    vintages converge toward it, so point-in-time selection has something to bite on.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for spec in config.MACRO_UNIVERSE:
        freq = "QS" if spec.freq == "quarterly" else "MS"
        periods = pd.date_range(
            config.START_DATE - pd.DateOffset(years=2), config.END_DATE, freq=freq
        )
        n = len(periods)
        if spec.name == "cpi":
            final = 240.0 * np.exp(np.cumsum(rng.normal(0.0022, 0.0018, n)))
            revision_sd = 0.12
        elif spec.name == "payrolls":
            final = 144_000 + np.cumsum(rng.normal(150.0, 260.0, n))
            revision_sd = 45.0
        elif spec.name == "unrate":
            final = np.clip(4.9 + np.cumsum(rng.normal(-0.005, 0.13, n)), 3.2, 11.0)
            revision_sd = 0.05
        elif spec.name == "indpro":
            final = 100.0 * np.exp(np.cumsum(rng.normal(0.0012, 0.0055, n)))
            revision_sd = 0.30
        else:  # real GDP
            final = 18_000.0 * np.exp(np.cumsum(rng.normal(0.0055, 0.0090, n)))
            revision_sd = 26.0

        for i, period in enumerate(periods):
            months = 1 if spec.freq == "monthly" else 3
            period_end = period + pd.DateOffset(months=months) - pd.Timedelta(days=1)
            first_release = _next_business_day(period_end + pd.Timedelta(days=spec.release_lag_days))
            # Three vintages: first print, first revision, final.
            errors = rng.normal(0.0, revision_sd, 2)
            schedule = [
                (first_release, final[i] + errors[0] + errors[1]),
                (_next_business_day(first_release + pd.DateOffset(months=1)), final[i] + errors[1]),
                (_next_business_day(first_release + pd.DateOffset(months=3)), final[i]),
            ]
            for vintage_date, value in schedule:
                if vintage_date > config.END_DATE:
                    continue
                rows.append(
                    {
                        "series": spec.name,
                        "reference_period": period,
                        "vintage_date": vintage_date,
                        "value": float(value),
                    }
                )
    frame = pd.DataFrame(rows).sort_values(["series", "reference_period", "vintage_date"])
    return frame.reset_index(drop=True)


def _next_business_day(ts: pd.Timestamp) -> pd.Timestamp:
    while ts.weekday() >= 5:
        ts = ts + pd.Timedelta(days=1)
    return ts


def build_cot(daily: pd.DataFrame, seed: int = config.SYNTHETIC_SEED + 2) -> pd.DataFrame:
    """Weekly Commitments of Traders positions in the platform's tidy COT schema.

    Columns: market, report, contract_name, report_date, category, long, short,
    open_interest. The live CFTC clients map their (differently named, wide) columns
    into this same schema, so downstream code has one shape to handle.

    Speculative net positioning is driven by trailing 20-day momentum in the market's
    price proxy plus a mean-reverting flow component. The remaining categories absorb
    the other side, and a non-reportable category closes the open-interest identity.
    """
    rng = np.random.default_rng(seed)
    report_dates = trading_calendar.cot_report_dates()
    proxy = _return_proxies(daily)
    rows = []

    for market in config.COT_UNIVERSE:
        momentum = proxy[market.price_proxy].rolling(20).sum().reindex(report_dates).ffill().fillna(0.0)
        momentum_z = (momentum / momentum.std()).clip(-3, 3).to_numpy()
        n_weeks = len(report_dates)
        oi_path = COT_BASE_OPEN_INTEREST[market.name] * np.exp(_ou(rng.standard_normal(n_weeks), 60.0, 0.11))
        flow = _ou(rng.standard_normal(n_weeks), 26.0, 1.0)
        spec_ratio = np.clip(0.055 * momentum_z + 0.045 * flow, -0.32, 0.32)

        categories = config.COT_CATEGORIES[market.report]
        spec_category = config.COT_SPEC_CATEGORY[market.report]
        other_names = [c for c in categories if c != spec_category]
        weights = np.array([0.55, 0.30, 0.15])[: len(other_names)]
        weights = weights / weights.sum()

        for i, report_date in enumerate(report_dates):
            oi = float(oi_path[i])
            net_ratios = {spec_category: float(spec_ratio[i])}
            for name, w in zip(other_names, weights):
                net_ratios[name] = -spec_ratio[i] * w + 0.004 * rng.standard_normal()

            longs, shorts = {}, {}
            for name in categories:
                net = net_ratios[name] * oi
                gross = max(COT_GROSS_SHARE[market.report][name] * oi, abs(net) * 1.15)
                longs[name] = round((gross + net) / 2.0)
                shorts[name] = round((gross - net) / 2.0)
            longs["nonreportable"] = round(oi) - sum(longs.values())
            shorts["nonreportable"] = round(oi) - sum(shorts.values())

            for name in (*categories, "nonreportable"):
                rows.append(
                    {
                        "market": market.name,
                        "report": market.report,
                        "contract_name": market.contract_name,
                        "report_date": report_date,
                        "category": name,
                        "long": longs[name],
                        "short": shorts[name],
                        "open_interest": round(oi),
                    }
                )

    frame = pd.DataFrame(rows)
    frame["report_date"] = pd.to_datetime(frame["report_date"])
    return frame.sort_values(["market", "report_date", "category"]).reset_index(drop=True)


def _return_proxies(daily: pd.DataFrame) -> pd.DataFrame:
    """Return proxies for the five COT markets (see CotMarket.proxy_note)."""
    out = pd.DataFrame(index=daily.index)
    out["spx"] = np.log(daily["spx"]).diff()
    out["wti"] = np.log(daily["wti"]).diff()
    out["gold"] = np.log(daily["gold"]).diff()
    out["usd_short_proxy"] = -np.log(daily["usd"]).diff()
    out["zn_proxy"] = -config.ZN_MODIFIED_DURATION * daily["dgs10"].diff() / 100.0
    return out
