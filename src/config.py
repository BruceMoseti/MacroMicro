"""Frozen configuration for the Macro Markets Quantitative Research Platform.

Every date, series identifier, split boundary and numeric constant used anywhere in
the project is defined here exactly once. The research period is frozen so that any
number quoted from this project can be regenerated bit-for-bit.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------------------
# Research period (frozen)
# --------------------------------------------------------------------------------------
# 2016-08-15 is the first date of the 10-year daily S&P 500 window published by FRED.
# 2026-08-13 is the last observation used by this study.
START_DATE = pd.Timestamp("2016-08-15")
END_DATE = pd.Timestamp("2026-08-13")

# Train / validation / test boundaries are declared before any performance is observed.
TRAIN_START, TRAIN_END = START_DATE, pd.Timestamp("2022-12-31")
VALID_START, VALID_END = pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31")
TEST_START, TEST_END = pd.Timestamp("2025-01-01"), END_DATE

SPLITS: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {
    "train": (TRAIN_START, TRAIN_END),
    "validation": (VALID_START, VALID_END),
    "test": (TEST_START, TEST_END),
}

# --------------------------------------------------------------------------------------
# Conventions
# --------------------------------------------------------------------------------------
TRADING_DAYS_PER_YEAR = 252
BP_PER_PCT = 100.0  # FRED yields are stored in percent; 1 pct point = 100 bp.

# Modified duration used to convert a 10y yield change into an approximate note return.
# This is an explicit approximation, not an observed futures return series.
ZN_MODIFIED_DURATION = 7.8

HORIZONS = (1, 5, 20, 63, 252)
VOL_WINDOWS = (20, 63, 252)
ZSCORE_WINDOWS = (60, 252)
ROLLING_CORR_WINDOWS = (63, 252)
ROLLING_BETA_WINDOW = 252
COT_ZSCORE_WEEKS = 156  # three years of weekly observations

# Transaction cost grid (one-way, applied to traded notional). Sensitivity, not a
# sourced cost model.
COST_GRID_BPS = (0.0, 1.0, 2.0, 5.0)
HEADLINE_COST_BPS = 2.0

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"
CHART_DIR = OUTPUT_DIR / "charts"
TABLE_DIR = OUTPUT_DIR / "tables"
DOCS_DIR = ROOT / "docs"
EXCEL_PATH = OUTPUT_DIR / "macro_market_monitor.xlsx"

for _p in (DATA_RAW, DATA_PROCESSED, CHART_DIR, TABLE_DIR, DOCS_DIR):
    _p.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------------------
# Market universe
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Series:
    """One daily observable in the research universe."""

    name: str  # internal column name
    fred_id: str  # FRED series identifier
    asset_class: str
    unit: str  # "percent" | "index" | "usd" | "bp"
    label: str
    kind: str  # "price" -> log returns; "yield" -> basis point changes
    coverage_note: str = ""


DAILY_UNIVERSE: tuple[Series, ...] = (
    # ---- Rates -----------------------------------------------------------------------
    Series("dgs2", "DGS2", "rates", "percent", "2y Treasury yield", "yield"),
    Series("dgs10", "DGS10", "rates", "percent", "10y Treasury yield", "yield"),
    Series("dgs30", "DGS30", "rates", "percent", "30y Treasury yield", "yield"),
    Series("real10", "DFII10", "rates", "percent", "10y TIPS real yield", "yield"),
    Series("breakeven10", "T10YIE", "rates", "percent", "10y breakeven inflation", "yield"),
    Series("fedfunds", "DFF", "rates", "percent", "Effective fed funds rate", "yield"),
    # ---- FX --------------------------------------------------------------------------
    Series("usd", "DTWEXBGS", "fx", "index", "Nominal broad USD index (Jan-2006=100)", "price"),
    # ---- Equities --------------------------------------------------------------------
    Series("spx", "SP500", "equity", "index", "S&P 500 close", "price"),
    Series("ndx", "NASDAQ100", "equity", "index", "Nasdaq 100 close", "price"),
    # ---- Volatility ------------------------------------------------------------------
    Series("vix", "VIXCLS", "volatility", "index", "Cboe VIX close", "price"),
    # ---- Commodities -----------------------------------------------------------------
    Series("wti", "DCOILWTICO", "commodity", "usd", "WTI crude, Cushing OK ($/bbl)", "price"),
    Series(
        "gold",
        "GOLDPMGBD228NLBM",
        "commodity",
        "usd",
        "LBMA gold price, PM fix ($/oz)",
        "price",
        coverage_note=(
            "FRED's LBMA gold series was discontinued; a live run needs an alternate "
            "gold source for the tail of the window (see docs/DATA_SOURCES.md)."
        ),
    ),
    # ---- Credit / financial conditions -----------------------------------------------
    Series("hy_oas", "BAMLH0A0HYM2", "credit", "percent", "ICE BofA US HY OAS", "yield"),
)

DAILY_COLUMNS = tuple(s.name for s in DAILY_UNIVERSE)
SERIES_BY_NAME = {s.name: s for s in DAILY_UNIVERSE}
YIELD_COLUMNS = tuple(s.name for s in DAILY_UNIVERSE if s.kind == "yield")
PRICE_COLUMNS = tuple(s.name for s in DAILY_UNIVERSE if s.kind == "price")

# Curve definitions, in percent (converted to bp where reported).
CURVE_DEFS = {"curve_2s10s": ("dgs10", "dgs2"), "curve_2s30s": ("dgs30", "dgs2")}


# --------------------------------------------------------------------------------------
# Vintage macro universe (ALFRED). release_lag_days is the conservative fallback used
# only when a vintage-capable API key is unavailable.
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class MacroSeries:
    name: str
    fred_id: str
    freq: str  # "monthly" | "quarterly"
    label: str
    release_lag_days: int


MACRO_UNIVERSE: tuple[MacroSeries, ...] = (
    MacroSeries("cpi", "CPIAUCSL", "monthly", "CPI, all urban consumers (SA)", 14),
    MacroSeries("payrolls", "PAYEMS", "monthly", "Nonfarm payrolls (SA, thousands)", 6),
    MacroSeries("unrate", "UNRATE", "monthly", "Unemployment rate (SA, %)", 6),
    MacroSeries("indpro", "INDPRO", "monthly", "Industrial production index", 16),
    MacroSeries("gdp", "GDPC1", "quarterly", "Real GDP (SAAR, chained 2017$)", 29),
)

# --------------------------------------------------------------------------------------
# CFTC Commitments of Traders universe
# --------------------------------------------------------------------------------------
# report: "tff"     -> Traders in Financial Futures (financial contracts)
#         "disagg"  -> Disaggregated (physical commodities)
# The speculative category differs by report: leveraged funds vs managed money.


@dataclass(frozen=True)
class CotMarket:
    name: str
    cme_code: str
    report: str
    contract_name: str  # CFTC "market_and_exchange_names" value
    label: str
    price_proxy: str  # column in the daily panel used as the return proxy
    proxy_note: str


COT_UNIVERSE: tuple[CotMarket, ...] = (
    CotMarket(
        "es",
        "ES",
        "tff",
        "E-MINI S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE",
        "E-mini S&P 500",
        "spx",
        "Cash S&P 500 log return used as the return proxy; not the ES futures return.",
    ),
    CotMarket(
        "zn",
        "ZN",
        "tff",
        "10-YEAR U.S. TREASURY NOTES - CHICAGO BOARD OF TRADE",
        "10y Treasury note",
        "zn_proxy",
        "Duration-approximated note return (-D x dy) with D=7.8; not the ZN futures return.",
    ),
    CotMarket(
        "6e",
        "6E",
        "tff",
        "EURO FX - CHICAGO MERCANTILE EXCHANGE",
        "Euro FX",
        "usd_short_proxy",
        "Negated broad USD index return as a long-EUR proxy; not the 6E futures return.",
    ),
    CotMarket(
        "cl",
        "CL",
        "disagg",
        "CRUDE OIL, LIGHT SWEET-WTI - NEW YORK MERCANTILE EXCHANGE",
        "WTI crude oil",
        "wti",
        "WTI spot log return used as the return proxy; excludes futures roll yield.",
    ),
    CotMarket(
        "gc",
        "GC",
        "disagg",
        "GOLD - COMMODITY EXCHANGE INC.",
        "Gold",
        "gold",
        "Gold spot log return used as the return proxy; excludes futures roll yield.",
    ),
)

COT_CATEGORIES = {
    "tff": ("dealer", "asset_manager", "leveraged_funds", "other_rept"),
    "disagg": ("prod_merc", "swap_dealer", "managed_money", "other_rept"),
}
# The category treated as "speculative crowding" for each report type.
COT_SPEC_CATEGORY = {"tff": "leveraged_funds", "disagg": "managed_money"}

# CFTC publication convention: positions are as of Tuesday's close and are released
# the following Friday at 15:30 US/Eastern.
COT_AS_OF_WEEKDAY = 1  # Tuesday
COT_RELEASE_WEEKDAY = 4  # Friday
COT_RELEASE_TIME = "15:30"
COT_RELEASE_TZ = "US/Eastern"

# --------------------------------------------------------------------------------------
# Real observations supplied as project anchors. These are checked by
# validation.check_anchor_values on every run.
# --------------------------------------------------------------------------------------
ANCHORS: dict[str, float] = {"spx": 7798.99, "dgs2": 4.15, "dgs10": 4.63}
ANCHOR_DATE = END_DATE
ANCHOR_CURVE_2S10S_BP = 48.0
ANCHOR_TOLERANCE = {"spx": 0.01, "dgs2": 0.005, "dgs10": 0.005}

# --------------------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------------------
SOURCE_LIVE = "LIVE"
SOURCE_SYNTHETIC = "SYNTHETIC"

DATASETS = (
    "FRED daily market series",
    "ALFRED vintage macroeconomic series",
    "CFTC Traders in Financial Futures (COT)",
    "CFTC Disaggregated Futures (COT)",
    "Cboe VIX history",
)

FRED_API_KEY = os.environ.get("FRED_API_KEY")
SYNTHETIC_SEED = 20260813


def split_of(index: pd.DatetimeIndex) -> pd.Series:
    """Label each timestamp with its train/validation/test split."""
    out = pd.Series("out_of_range", index=index, dtype="object")
    for label, (lo, hi) in SPLITS.items():
        out[(index >= lo) & (index <= hi)] = label
    return out


def slice_split(frame: pd.DataFrame | pd.Series, label: str):
    """Restrict a time-indexed object to one declared split."""
    lo, hi = SPLITS[label]
    return frame.loc[(frame.index >= lo) & (frame.index <= hi)]
