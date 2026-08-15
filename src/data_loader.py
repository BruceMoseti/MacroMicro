"""Ingestion layer: FRED, ALFRED, CFTC and Cboe, with a frozen synthetic fallback.

Raw payloads are written to ``data/raw/<source>/`` exactly as received and are never
mutated in place. Everything downstream reads the normalised frames returned here.

Two source modes:

``--source fred``       Live download. Requires network egress to fred.stlouisfed.org,
                        publicreporting.cftc.gov and cdn.cboe.com. Vintage macro data
                        additionally requires ``FRED_API_KEY``; without it the loader
                        falls back to a release-lag approximation and records that
                        downgrade in the provenance record.
``--source synthetic``  Frozen seeded dataset from ``src/synthetic.py``. Same schema,
                        calendar and units. Used when egress is unavailable.

The live endpoints below could not be exercised from the environment this project was
built in (all market-data hosts were blocked at the network layer), so the request
shapes follow each provider's documented API and should be confirmed on a first live
run. The synthetic path is fully exercised by the test suite.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import requests

from . import config, synthetic

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
CBOE_VIX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
CFTC_SOCRATA_URL = "https://publicreporting.cftc.gov/resource/{dataset}.json"

# CFTC Public Reporting Environment dataset identifiers. Verify against the live
# catalogue at https://publicreporting.cftc.gov on first use.
CFTC_DATASETS = {"tff": "gpe5-46if", "disagg": "72hh-3qpy"}

# Socrata column names -> platform category names.
CFTC_COLUMN_MAP = {
    "tff": {
        "dealer": ("dealer_positions_long_all", "dealer_positions_short_all"),
        "asset_manager": ("asset_mgr_positions_long", "asset_mgr_positions_short"),
        "leveraged_funds": ("lev_money_positions_long", "lev_money_positions_short"),
        "other_rept": ("other_rept_positions_long", "other_rept_positions_short"),
        "nonreportable": ("nonrept_positions_long_all", "nonrept_positions_short_all"),
    },
    "disagg": {
        "prod_merc": ("prod_merc_positions_long", "prod_merc_positions_short"),
        "swap_dealer": ("swap_positions_long_all", "swap__positions_short_all"),
        "managed_money": ("m_money_positions_long_all", "m_money_positions_short_all"),
        "other_rept": ("other_rept_positions_long", "other_rept_positions_short"),
        "nonreportable": ("nonrept_positions_long_all", "nonrept_positions_short_all"),
    },
}
REQUEST_TIMEOUT = 60


@dataclass
class Dataset:
    """Everything the research layer needs, plus how it was obtained."""

    daily: pd.DataFrame
    macro_vintages: pd.DataFrame
    cot: pd.DataFrame
    source: str
    notes: list[str] = field(default_factory=list)

    def provenance(self) -> dict:
        return {
            "source": self.source,
            "is_synthetic": self.source == config.SOURCE_SYNTHETIC,
            "start_date": str(self.daily.index.min().date()),
            "end_date": str(self.daily.index.max().date()),
            "daily_observations": int(len(self.daily)),
            "daily_series": int(self.daily.shape[1]),
            "macro_series": int(self.macro_vintages["series"].nunique()),
            "macro_vintage_rows": int(len(self.macro_vintages)),
            "cot_markets": int(self.cot["market"].nunique()),
            "cot_report_dates": int(self.cot["report_date"].nunique()),
            "notes": self.notes,
        }


# --------------------------------------------------------------------------------------
# FRED / ALFRED
# --------------------------------------------------------------------------------------
def _fred_series(series_id: str, raw_dir: Path) -> pd.Series:
    """One daily FRED series over the frozen window, in its published units."""
    params = {
        "id": series_id,
        "cosd": config.START_DATE.strftime("%Y-%m-%d"),
        "coed": config.END_DATE.strftime("%Y-%m-%d"),
    }
    response = requests.get(FRED_CSV_URL, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    (raw_dir / f"fred_{series_id}.csv").write_bytes(response.content)

    frame = pd.read_csv(io.BytesIO(response.content))
    date_column = frame.columns[0]  # "observation_date" or the legacy "DATE"
    frame[date_column] = pd.to_datetime(frame[date_column])
    # FRED encodes missing observations as ".".
    values = pd.to_numeric(frame[frame.columns[1]], errors="coerce")
    return pd.Series(values.to_numpy(), index=pd.DatetimeIndex(frame[date_column]), name=series_id)


def _fred_daily_panel(raw_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    columns, notes = {}, []
    for spec in config.DAILY_UNIVERSE:
        series = _fred_series(spec.fred_id, raw_dir)
        columns[spec.name] = series
        if spec.coverage_note:
            notes.append(f"{spec.name} ({spec.fred_id}): {spec.coverage_note}")
    panel = pd.DataFrame(columns)
    panel.index.name = "date"
    return panel.sort_index(), notes


def _alfred_vintages(raw_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """Point-in-time macro data.

    With an API key this returns the true ALFRED vintage table: every value tagged with
    the date it first became available. Without one, the current (revised) series is
    returned with a conservative release lag, which removes publication-lag leakage but
    not revision bias. The downgrade is recorded so it appears in the report.
    """
    if not config.FRED_API_KEY:
        notes = [
            "FRED_API_KEY not set: macro data uses a release-lag approximation, not true "
            "ALFRED vintages. Publication-lag leakage is removed; revision bias is not."
        ]
        rows = []
        for spec in config.MACRO_UNIVERSE:
            series = _fred_series(spec.fred_id, raw_dir).dropna()
            months = 1 if spec.freq == "monthly" else 3
            for period, value in series.items():
                period_end = period + pd.DateOffset(months=months) - pd.Timedelta(days=1)
                rows.append(
                    {
                        "series": spec.name,
                        "reference_period": period,
                        "vintage_date": period_end + pd.Timedelta(days=spec.release_lag_days),
                        "value": float(value),
                    }
                )
        return pd.DataFrame(rows), notes

    rows = []
    for spec in config.MACRO_UNIVERSE:
        params = {
            "series_id": spec.fred_id,
            "api_key": config.FRED_API_KEY,
            "file_type": "json",
            "realtime_start": "1776-07-04",
            "realtime_end": "9999-12-31",
            "output_type": 1,
        }
        response = requests.get(FRED_API_URL, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        (raw_dir / f"alfred_{spec.fred_id}.json").write_bytes(response.content)
        for observation in response.json()["observations"]:
            if observation["value"] == ".":
                continue
            rows.append(
                {
                    "series": spec.name,
                    "reference_period": pd.Timestamp(observation["date"]),
                    "vintage_date": pd.Timestamp(observation["realtime_start"]),
                    "value": float(observation["value"]),
                }
            )
    return pd.DataFrame(rows), ["Macro data uses true ALFRED vintages (realtime_start as vintage date)."]


# --------------------------------------------------------------------------------------
# CFTC Commitments of Traders
# --------------------------------------------------------------------------------------
def _cftc_report(report: str, raw_dir: Path) -> pd.DataFrame:
    """One CFTC report type, normalised into the platform's tidy COT schema."""
    markets = [m for m in config.COT_UNIVERSE if m.report == report]
    contract_filter = " OR ".join(f"market_and_exchange_names='{m.contract_name}'" for m in markets)
    params = {
        "$where": (
            f"report_date_as_yyyy_mm_dd >= '{config.START_DATE:%Y-%m-%d}' "
            f"AND report_date_as_yyyy_mm_dd <= '{config.END_DATE:%Y-%m-%d}' AND ({contract_filter})"
        ),
        "$limit": 50_000,
    }
    url = CFTC_SOCRATA_URL.format(dataset=CFTC_DATASETS[report])
    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    (raw_dir / f"cftc_{report}.json").write_bytes(response.content)

    raw = pd.DataFrame(response.json())
    by_contract = {m.contract_name: m for m in markets}
    rows = []
    for _, record in raw.iterrows():
        market = by_contract[record["market_and_exchange_names"]]
        open_interest = float(record["open_interest_all"])
        for category, (long_column, short_column) in CFTC_COLUMN_MAP[report].items():
            rows.append(
                {
                    "market": market.name,
                    "report": report,
                    "contract_name": market.contract_name,
                    "report_date": pd.Timestamp(record["report_date_as_yyyy_mm_dd"]),
                    "category": category,
                    "long": float(record[long_column]),
                    "short": float(record[short_column]),
                    "open_interest": open_interest,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Cboe (cross-source consistency check on VIX)
# --------------------------------------------------------------------------------------
def load_cboe_vix(raw_dir: Path | None = None) -> pd.Series:
    """Cboe's own VIX close, used to cross-check the FRED VIX series."""
    raw_dir = raw_dir or (config.DATA_RAW / config.SOURCE_LIVE.lower())
    raw_dir.mkdir(parents=True, exist_ok=True)
    response = requests.get(CBOE_VIX_URL, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    (raw_dir / "cboe_vix_history.csv").write_bytes(response.content)
    frame = pd.read_csv(io.BytesIO(response.content))
    frame["DATE"] = pd.to_datetime(frame["DATE"])
    series = frame.set_index("DATE")["CLOSE"].sort_index()
    return series.loc[config.START_DATE : config.END_DATE].rename("vix_cboe")


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------
def load(source: str = config.SOURCE_SYNTHETIC) -> Dataset:
    """Build the normalised dataset from ``source`` and persist the raw layer."""
    raw_dir = config.DATA_RAW / source.lower()
    raw_dir.mkdir(parents=True, exist_ok=True)

    if source == config.SOURCE_SYNTHETIC:
        daily = synthetic.build_daily_panel()
        macro = synthetic.build_macro_vintages()
        cot = synthetic.build_cot(daily)
        notes = [
            "SYNTHETIC DATA. Generated by src/synthetic.py; not observed market data.",
            synthetic.SCENARIO_NOTE,
            f"Anchored to the {config.ANCHOR_DATE:%Y-%m-%d} observations in config.ANCHORS.",
        ]
        daily.to_csv(raw_dir / "daily_panel.csv")
        macro.to_csv(raw_dir / "macro_vintages.csv", index=False)
        cot.to_csv(raw_dir / "cot.csv", index=False)
    elif source == config.SOURCE_LIVE:
        daily, notes = _fred_daily_panel(raw_dir)
        macro, macro_notes = _alfred_vintages(raw_dir)
        notes = notes + macro_notes
        cot = pd.concat([_cftc_report(report, raw_dir) for report in ("tff", "disagg")], ignore_index=True)
    else:
        raise ValueError(f"unknown source {source!r}; expected {config.SOURCE_LIVE} or {config.SOURCE_SYNTHETIC}")

    daily = daily.loc[config.START_DATE : config.END_DATE]
    macro = macro.sort_values(["series", "reference_period", "vintage_date"]).reset_index(drop=True)
    cot = cot.sort_values(["market", "report_date", "category"]).reset_index(drop=True)

    dataset = Dataset(daily=daily, macro_vintages=macro, cot=cot, source=source, notes=notes)
    (config.DATA_PROCESSED / "provenance.json").write_text(json.dumps(dataset.provenance(), indent=2) + "\n")
    (raw_dir / "_PROVENANCE.txt").write_text("\n".join(notes) + "\n")
    return dataset
