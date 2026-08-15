#!/usr/bin/env python3
"""Run the full research pipeline.

    python run_pipeline.py                    # frozen synthetic dataset (no network needed)
    python run_pipeline.py --source fred      # live FRED / ALFRED / CFTC / Cboe download

Writes data/processed, outputs/charts, outputs/tables, outputs/macro_market_monitor.xlsx,
docs/RESEARCH_REPORT.md and docs/KEY_FIGURES.md. Exits non-zero if any critical
validation check fails.
"""

from __future__ import annotations

import argparse
import sys

from src import config, pipeline, validation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source",
        choices=[config.SOURCE_SYNTHETIC.lower(), "fred"],
        default=config.SOURCE_SYNTHETIC.lower(),
        help="synthetic (default) uses the frozen seeded dataset; fred downloads live data",
    )
    parser.add_argument("--no-outputs", action="store_true", help="run the research only, write nothing")
    args = parser.parse_args()

    source = config.SOURCE_LIVE if args.source == "fred" else config.SOURCE_SYNTHETIC
    print("Macro Markets Quantitative Research Platform")
    print(f"  window     {config.START_DATE:%Y-%m-%d} to {config.END_DATE:%Y-%m-%d} (frozen)")
    print(f"  source     {source}")

    try:
        results = pipeline.run(source=source, write_outputs=not args.no_outputs)
    except validation.ValidationError as error:
        print(f"\nPIPELINE FAILED\n{error}", file=sys.stderr)
        return 1

    log = results["validation_log"]
    provenance = results["dataset"].provenance()
    headline = results["backtest_headline"]
    test = headline[headline["split"] == "test"].sort_values("sharpe", ascending=False)

    print(f"\n  observations   {provenance['daily_observations']:,} trading days")
    print(f"  features       {results['features'].shape[1]}")
    print(f"  signals        {len(results['signal_registry'])} candidates, "
          f"{int(results['signal_registry']['backtested'].sum())} backtested")
    print(f"  trades         {int(headline[headline['split'] == 'full']['trades'].sum()):,} full sample")
    print(f"  validation     {log.summary()}")
    if len(test):
        best = test.iloc[0]
        print(f"  best OOS       {best['signal']} Sharpe {best['sharpe']:.2f} "
              f"({int(best['trades'])} trades, {config.HEADLINE_COST_BPS:.0f}bp)")
    if not args.no_outputs:
        print(f"\n  charts         {len(results.get('chart_manifest', []))} -> {config.CHART_DIR}")
        print(f"  tables         -> {config.TABLE_DIR}")
        print(f"  monitor        -> {config.EXCEL_PATH}")
        print(f"  report         -> {config.DOCS_DIR / 'RESEARCH_REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
