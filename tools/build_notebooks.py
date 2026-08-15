#!/usr/bin/env python3
"""Generate and execute the research notebooks.

The notebooks are generated from this script so their code cannot drift away from the
library. Each one calls ``pipeline.run(write_outputs=False)`` and inspects the same
objects the validation checks ran against.

    python tools/build_notebooks.py            # write and execute
    python tools/build_notebooks.py --no-exec  # write only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"

PREAMBLE = """\
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent))

import numpy as np
import pandas as pd

from src import backtest, config, features, pipeline, regressions, relative_value, signals

# src.charts selects the Agg backend so the pipeline runs headless; switch back to the
# inline backend afterwards so figures render in the notebook.
%matplotlib inline
import matplotlib.pyplot as plt

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

results = pipeline.run(write_outputs=False)
provenance = results["dataset"].provenance()
print(f"source        {provenance['source']}")
print(f"window        {provenance['start_date']} to {provenance['end_date']}")
print(f"observations  {provenance['daily_observations']:,} trading days")
print(f"validation    {results['validation_log'].summary()}")
"""

PROVENANCE_WARNING = """\
> **Data provenance.** This notebook runs on whichever source the pipeline was given. When the
> source is `SYNTHETIC`, every number below is a property of the simulator in `src/synthetic.py`
> and **not** market history. The only observed market values in the project are the three
> anchors in `config.ANCHORS`. Run `python run_pipeline.py --source fred` on a machine with
> network access to reproduce the identical analysis from live data.
"""


def notebook(cells: list[tuple[str, str]]) -> nbf.NotebookNode:
    book = nbf.v4.new_notebook()
    book.cells = [
        nbf.v4.new_markdown_cell(body) if kind == "md" else nbf.v4.new_code_cell(body) for kind, body in cells
    ]
    book.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    return book


# --------------------------------------------------------------------------------------
NOTEBOOKS: dict[str, list[tuple[str, str]]] = {
    "01_market_overview.ipynb": [
        (
            "md",
            "# 1. Market overview\n\n"
            "The frozen research window, the instrument universe, and the anchor check that every "
            "run has to pass before anything else happens.\n\n" + PROVENANCE_WARNING,
        ),
        ("code", PREAMBLE),
        (
            "md",
            "## The anchor check\n\n"
            "The project is anchored to three observed market values on 2026-08-13. If the loaded "
            "data does not reproduce them, the pipeline aborts rather than producing plausible "
            "nonsense. This is the cheapest possible guard against silently loading the wrong "
            "series or the wrong window.",
        ),
        (
            "code",
            'panel = results["panel"]\n'
            "row = panel.loc[config.ANCHOR_DATE]\n"
            'for name, expected in config.ANCHORS.items():\n'
            '    print(f"{name:6s} actual {float(row[name]):>10,.2f}   expected {expected:>10,.2f}")\n'
            'curve_bp = (row["dgs10"] - row["dgs2"]) * config.BP_PER_PCT\n'
            'print(f"\\n2s10s = {row[\'dgs10\']:.2f}% - {row[\'dgs2\']:.2f}% = {curve_bp:+.0f}bp "\n'
            '      f"(expected {config.ANCHOR_CURVE_2S10S_BP:+.0f}bp)")',
        ),
        (
            "md",
            "## Universe and latest levels\n\n"
            "Every series carries its own last observation date, so a stale input cannot be compared "
            "against a fresh one without it being visible.",
        ),
        (
            "code",
            "from src import reporting\n\n"
            'snapshot = reporting.market_snapshot(panel, results["curves"])\n'
            'snapshot[["series", "label", "asset_class", "fred_id", "latest", "last_observation", "observations", "missing"]]',
        ),
        ("md", "## Cross-asset changes over standard horizons"),
        (
            "code",
            'changes = reporting.cross_asset_changes(panel, results["curves"])\n'
            'changes[["series", "label", "unit", "latest", "1d", "1w", "1m", "3m", "12m"]]',
        ),
        ("md", "## The cycle in one picture\n\nPolicy, the curve, and the inversion episode."),
        (
            "code",
            "fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)\n"
            'for column, label in (("dgs2", "2y"), ("dgs10", "10y"), ("dgs30", "30y")):\n'
            "    axes[0].plot(panel.index, panel[column], linewidth=1.0, label=label)\n"
            'axes[0].set_ylabel("yield (%)"); axes[0].legend(ncol=3); axes[0].grid(alpha=0.3)\n'
            'axes[0].set_title("Treasury yields")\n\n'
            'curve = results["curves"]["curve_2s10s"]\n'
            'axes[1].plot(curve.index, curve, linewidth=1.0, color="#1f4e79")\n'
            'axes[1].axhline(0, color="#c00000")\n'
            'axes[1].fill_between(curve.index, curve, 0, where=curve < 0, color="#c00000", alpha=0.2)\n'
            'axes[1].set_ylabel("2s10s (bp)"); axes[1].grid(alpha=0.3)\n'
            'axes[1].set_title(f"2s10s: inverted on {int((curve < 0).sum())} of {len(curve)} sessions")\n'
            "plt.tight_layout(); plt.show()",
        ),
        (
            "md",
            "## Regime classification\n\n"
            "Four dimensions with fixed rules, so \"market conditions\" is a label with a day count "
            "rather than an adjective. These are ex-post attribution buckets, not tradable filters: "
            "the volatility and real-yield breakpoints use full-sample quantiles.",
        ),
        ("code", 'results["regime_summary"]'),
        (
            "md",
            "## Validation\n\n"
            "Every check that ran on this execution. Critical failures abort the pipeline; advisory "
            "rows document known data limitations instead of hiding them.",
        ),
        (
            "code",
            'log = results["validation"]\n'
            'print(log["result"].value_counts().to_string())\n'
            'log[log["check"].str.startswith(("daily", "anchors", "cot", "macro", "futures"))]',
        ),
    ],
    "02_cross_asset_relationships.ipynb": [
        (
            "md",
            "# 2. Cross-asset relationships (Hypothesis 1)\n\n"
            "**Question.** Do changes in Treasury yields, the curve, real yields and implied "
            "volatility explain short-term equity returns, and is the relationship stable?\n\n"
            + PROVENANCE_WARNING,
        ),
        ("code", PREAMBLE),
        (
            "md",
            "## The specification cannot be estimated as usually written\n\n"
            "The natural specification is\n\n"
            "$$r_{SPX,t} = \\alpha + \\beta_1 \\Delta y_{2,t} + \\beta_2 \\Delta y_{10,t} "
            "+ \\beta_3 \\Delta Curve_t + \\beta_4 \\Delta VIX_t + \\epsilon_t$$\n\n"
            "and it is exactly rank deficient, because $\\Delta Curve_t \\equiv \\Delta y_{10,t} - "
            "\\Delta y_{2,t}$. The design matrix is singular, so the individual coefficients are not "
            "identified. Estimating it anyway and reading the condition number is the fastest way to "
            "see the problem.",
        ),
        (
            "code",
            'features_frame = results["features"]\n'
            "rows = []\n"
            "for name, regressors in regressions.SPECIFICATIONS.items():\n"
            "    _, diagnostics = regressions.fit_ols(features_frame, regressors, label=name)\n"
            "    rows.append(diagnostics)\n"
            'diag = pd.DataFrame(rows).set_index("spec")\n'
            'diag[["nobs", "r_squared", "adj_r_squared", "condition_number", "max_vif", "rank_deficient"]]',
        ),
        (
            "md",
            "The `naive` and `slope` specifications have identical R-squared because they span the "
            "same space; only `slope` has identified coefficients. The `real` specification uses the "
            "Fisher decomposition $y_{10} = r_{10} + \\pi_{10}$ to separate a discount-rate shock "
            "from an inflation-expectations shock, which raises explanatory power because the two "
            "have opposite signs for equities.",
        ),
        (
            "code",
            'coefficients = results["h1_coefficients"]\n'
            'coefficients[coefficients["spec"] == "real"][["term", "coefficient", "std_error_hac", "t_stat", "p_value", "nobs"]]',
        ),
        (
            "md",
            "## Why HAC standard errors\n\n"
            "Breusch-Pagan rejects homoskedasticity and Ljung-Box rejects zero residual "
            "autocorrelation, so ordinary standard errors would be too small and every t-statistic "
            "too flattering. Newey-West with the standard $\\lfloor 4(n/100)^{2/9} \\rfloor$ "
            "bandwidth is used throughout.",
        ),
        (
            "code",
            'diag[["hac_lags", "durbin_watson", "breusch_pagan_p", "ljung_box_p_10", "residual_skew", "residual_kurtosis"]]',
        ),
        (
            "md",
            "## Is the relationship stable?\n\n"
            "This is the part full-sample coefficients hide.",
        ),
        (
            "code",
            'rolling = results["h1_rolling_betas"]\n'
            "fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)\n"
            'axes[0].plot(rolling.index, rolling["beta_dy_1d_real10"] * 100, label="10y real yield", color="#1f4e79")\n'
            'axes[0].plot(rolling.index, rolling["beta_dy_1d_breakeven10"] * 100, label="10y breakeven", color="#e07b00")\n'
            'axes[0].axhline(0, color="#c00000"); axes[0].legend(); axes[0].grid(alpha=0.3)\n'
            'axes[0].set_ylabel("% per bp"); axes[0].set_title("Rolling 252d equity sensitivity to rate shocks")\n'
            'axes[1].plot(rolling.index, rolling["beta_ret_1d_vix"], color="#c00000")\n'
            'axes[1].axhline(0, color="#333333"); axes[1].grid(alpha=0.3)\n'
            'axes[1].set_ylabel("beta to VIX"); axes[1].set_title("Rolling 252d equity sensitivity to VIX")\n'
            "plt.tight_layout(); plt.show()\n\n"
            'results["h1_stability"]',
        ),
        (
            "md",
            "The VIX coefficient never changes sign. The real-yield coefficient does, which means "
            "the full-sample estimate is averaging over two different regimes rather than measuring "
            "one structural relationship.",
        ),
        (
            "md",
            "## Explanatory power is not predictive power\n\n"
            "`contemporaneous` fits on the training split and measures R-squared on later splits "
            "with same-day regressors: it tests whether the relationship is stable. It is not "
            "tradable, because same-day yield and VIX changes are not known when the equity return "
            "is realised. `predictive` lags every regressor one day, which is the version you could "
            "actually trade.",
        ),
        ("code", 'results["h1_out_of_sample"]'),
        (
            "md",
            "**Conclusion.** Hypothesis 1 is a strong contemporaneous relationship with correct "
            "signs and no forecasting content. Reporting the contemporaneous R-squared as though it "
            "were predictive would be the single most misleading thing this project could do.",
        ),
        (
            "md",
            "## Rolling correlations across the cross-asset set",
        ),
        (
            "code",
            'rolling_rel = results["rolling_relationships"]\n'
            "panels = [(\"equity_vs_rates\", \"S&P 500 vs 10y yield change\"),\n"
            "          (\"equity_vs_vix\", \"S&P 500 vs VIX change\"),\n"
            "          (\"usd_vs_wti\", \"Broad USD vs WTI\"),\n"
            "          (\"gold_vs_real\", \"Gold vs 10y real yield change\")]\n"
            "fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True, sharey=True)\n"
            "for ax, (key, title) in zip(axes.ravel(), panels):\n"
            "    for window in config.ROLLING_CORR_WINDOWS:\n"
            '        ax.plot(rolling_rel.index, rolling_rel[f"corr{window}_{key}"], linewidth=0.9, label=f"{window}d")\n'
            '    ax.axhline(0, color="#c00000"); ax.set_ylim(-1, 1); ax.set_title(title); ax.grid(alpha=0.3); ax.legend()\n'
            "plt.tight_layout(); plt.show()",
        ),
    ],
    "03_relative_value.ipynb": [
        (
            "md",
            "# 3. Cross-asset relative value (Hypothesis 2)\n\n"
            "**Question.** Do economically linked pairs produce residuals that mean-revert, and is "
            "the mean reversion tradable?\n\n" + PROVENANCE_WARNING,
        ),
        ("code", PREAMBLE),
        (
            "md",
            "## Pairs were chosen from economics, not from Sharpe ratios\n\n"
            "Each pair carries the prior that motivated it. The primary pair is the one with the "
            "cleanest theoretical link, fixed before any backtest was run.",
        ),
        (
            "code",
            "for pair in relative_value.PAIRS:\n"
            '    marker = " (PRIMARY)" if pair.name == relative_value.PRIMARY_PAIR else ""\n'
            '    print(f"{pair.name}{marker}\\n  {pair.y} ~ {pair.x}\\n  {pair.rationale}\\n")',
        ),
        (
            "md",
            "## The spurious regression\n\n"
            "Two trending near-unit-root series can produce a large, significant, and economically "
            "wrong beta in levels. Comparing the levels and differenced estimates makes it visible "
            "instead of leaving it buried.",
        ),
        (
            "code",
            'rv = results["rv_relationships"]\n'
            'rv[["pair", "beta", "beta_t", "r_squared", "beta_changes", "beta_changes_t",\n'
            '    "r_squared_changes", "corr_changes", "sign_agrees_with_changes"]]',
        ),
        (
            "code",
            "import matplotlib.dates as mdates\n\n"
            'panel = results["panel"]\n'
            'row = rv.set_index("pair").loc[relative_value.PRIMARY_PAIR]\n'
            'log_gold, real = np.log(panel["gold"]), panel["real10"]\n'
            "fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5))\n"
            'sc = left.scatter(real, log_gold, c=mdates.date2num(panel.index), cmap="viridis", s=4)\n'
            "grid = np.linspace(real.min(), real.max(), 50)\n"
            'left.plot(grid, row["alpha"] + row["beta"] * grid, color="#c00000", linewidth=2)\n'
            'left.set_xlabel("10y real yield (%)"); left.set_ylabel("log gold"); left.grid(alpha=0.3)\n'
            'left.set_title(f"Levels: beta = {row[\'beta\']:+.4f} (t = {row[\'beta_t\']:.1f})")\n'
            'bar = fig.colorbar(sc, ax=left); bar.ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y"))\n'
            'ch = pd.concat({"dy": log_gold.diff(), "dx": real.diff()}, axis=1).dropna()\n'
            'right.scatter(ch["dx"] * 100, ch["dy"] * 100, s=4, alpha=0.3, color="#1f4e79")\n'
            'grid = np.linspace(ch["dx"].min(), ch["dx"].max(), 50)\n'
            'right.plot(grid * 100, row["beta_changes"] * grid * 100, color="#c00000", linewidth=2)\n'
            'right.set_xlabel("d 10y real yield (bp)"); right.set_ylabel("gold return (%)"); right.grid(alpha=0.3)\n'
            'right.set_title(f"Changes: beta = {row[\'beta_changes\']:+.4f} (t = {row[\'beta_changes_t\']:.1f})")\n'
            "plt.tight_layout(); plt.show()",
        ),
        (
            "md",
            "The levels panel slopes the wrong way, and the colour bar shows why: the fit is tracking "
            "calendar time, because both series trended upward across the window. In differences the "
            "sign matches theory. Anyone reporting only the levels beta would have published a "
            "result with the wrong sign and a t-statistic large enough to look convincing.",
        ),
        (
            "md",
            "## Does the spread mean-revert?\n\n"
            "Tested on the **training split**, because that is the sample that existed when the "
            "strategy was specified. The plain ADF applied to a fitted residual is biased toward "
            "rejecting the unit root, so the Engle-Granger test, whose critical values account for "
            "the estimated cointegrating vector, is the one to believe.",
        ),
        (
            "code",
            'stats = results["rv_statistics"]\n'
            'stats[stats["sample"] == "train"][["pair", "nobs", "adf_stat", "adf_p", "engle_granger_p", "cointegrated_5pct", "ar1_phi", "half_life_days"]]',
        ),
        (
            "md",
            "No pair is cointegrated at 5%, and AR(1) persistence implies half-lives of hundreds of "
            "days: far too slow to trade. The residual from a *rolling* hedge ratio does reject the "
            "unit root with a much shorter half-life, but that is substantially mechanical, because a "
            "trailing regression re-centres its own residual every day. Both are reported so the "
            "difference is visible.",
        ),
        (
            "code",
            'stats[stats["sample"].isin(["full", "full_rolling_beta"])][["pair", "sample", "adf_p", "engle_granger_p", "ar1_phi", "half_life_days"]]',
        ),
        ("md", "## The tradable residual and its z-score"),
        (
            "code",
            'zscores, residuals = results["rv_zscores"], results["rv_residuals"]\n'
            "name = relative_value.PRIMARY_PAIR\n"
            "fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)\n"
            'axes[0].plot(residuals.index, residuals[name], linewidth=0.8, color="#1f4e79")\n'
            'axes[0].axhline(0, color="#999999"); axes[0].grid(alpha=0.3)\n'
            'axes[0].set_title(f"{name}: rolling-beta residual (252d trailing hedge ratio)")\n'
            "z = zscores[name]\n"
            'axes[1].plot(z.index, z, linewidth=0.7, color="#333333")\n'
            "for level in (signals.ENTRY_Z, -signals.ENTRY_Z):\n"
            '    axes[1].axhline(level, color="#c00000")\n'
            "for level in (signals.EXIT_Z, -signals.EXIT_Z):\n"
            '    axes[1].axhline(level, color="#2e7d32", linestyle="--")\n'
            'axes[1].grid(alpha=0.3); axes[1].set_ylabel("z (60d)")\n'
            'axes[1].set_title(f"Entry |z| > {signals.ENTRY_Z}, exit |z| < {signals.EXIT_Z}: "\n'
            '                  f"{int((z.abs() > signals.ENTRY_Z).sum())} sessions beyond the entry band")\n'
            "plt.tight_layout(); plt.show()",
        ),
        (
            "md",
            "## Why the hedge matters\n\n"
            "A residual is a two-legged position. For the primary pair the residual does converge "
            "after an extreme reading, but most of the convergence arrives through the real-yield "
            "leg, which is not investable in a cash universe. Trading gold alone leaves the position "
            "wearing gold's drift, which over this window is much larger than the convergence.",
        ),
        ("code", 'results["rv_convergence"]'),
        (
            "md",
            "The residual falls when it was rich and rises when it was cheap, so the mean reversion "
            "is there. But the tradable leg's return is positive in *both* states, because the "
            "unconditional drift dominates: most of the convergence arrives through the real-yield "
            "leg, which cannot be held in a cash universe. That is the whole argument for holding "
            "both legs, and it is why the hedged and single-leg variants are both backtested.",
        ),
        ("code", 'results["rv_betas"].describe().T[["mean", "std", "min", "max"]]'),
        (
            "md",
            "The hedge ratios are not stable either. `curve_hyoas` swings across hundreds of units, "
            "which is why it is analysed but never backtested: a spread whose hedge ratio is that "
            "unstable is not a spread.",
        ),
    ],
    "04_positioning.ipynb": [
        (
            "md",
            "# 4. Futures positioning (Hypothesis 3)\n\n"
            "**Question.** Does extreme speculative futures positioning predict continuation, "
            "reversal, or nothing?\n\n" + PROVENANCE_WARNING,
        ),
        ("code", PREAMBLE),
        (
            "md",
            "## The publication timing problem\n\n"
            "CFTC Commitments of Traders positions are measured at **Tuesday's close** and published "
            "the following **Friday at 15:30 US/Eastern**. Handing Tuesday's reading to a Wednesday "
            "model gives it three days of information that did not exist. Because 15:30 ET falls "
            "inside the cash session, this pipeline does not assume a fill in the closing half hour "
            "either: a report becomes usable on the next trading session.",
        ),
        (
            "code",
            'cot = results["dataset"].cot\n'
            'weekly = cot.drop_duplicates("report_date")[["report_date"]].reset_index(drop=True)\n'
            'weekly["available_from"] = features.cot_available_from(weekly["report_date"])\n'
            'weekly["as_of_day"] = weekly["report_date"].dt.day_name()\n'
            'weekly["usable_day"] = weekly["available_from"].dt.day_name()\n'
            'weekly["lag_days"] = (weekly["available_from"] - weekly["report_date"]).dt.days\n'
            "print(weekly.head(6).to_string(index=False))\n"
            'print("\\nlag distribution (7 days when that Monday is a holiday):")\n'
            'print(weekly["lag_days"].value_counts().sort_index().to_string())',
        ),
        (
            "md",
            "## Normalising positioning\n\n"
            "Raw contract counts are not comparable across markets or across time as open interest "
            "grows. The platform uses net position over open interest, then a three-year (156 week) "
            "trailing z-score, so a statement like *leveraged fund Treasury positioning is 2.1 "
            "standard deviations above its three-year mean* is well defined.",
        ),
        (
            "code",
            "from src import reporting\n\n"
            'snapshot = reporting.positioning_snapshot(results["features"], cot)\n'
            'snapshot[["market", "label", "category", "report_date_tuesday", "usable_from", "net_position", "open_interest", "net_pct_of_oi", "percentile_52w", "zscore_3y", "weekly_change_pct_oi"]]',
        ),
        (
            "code",
            'features_frame = results["features"]\n'
            "fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)\n"
            'ratio = features_frame["cot_zn_ratio"].dropna()\n'
            'axes[0].plot(ratio.index, ratio * 100, linewidth=1.0, color="#1f4e79")\n'
            'axes[0].axhline(0, color="#999999"); axes[0].grid(alpha=0.3)\n'
            'axes[0].set_ylabel("net % of OI")\n'
            'axes[0].set_title("10y Treasury note: leveraged fund net positioning, aligned to publication")\n'
            'z = features_frame["cot_zn_z156"].dropna()\n'
            'axes[1].plot(z.index, z, linewidth=1.0, color="#c00000")\n'
            "for level in (2, -2):\n"
            '    axes[1].axhline(level, color="#c00000", linestyle="--")\n'
            'axes[1].axhline(0, color="#999999"); axes[1].grid(alpha=0.3); axes[1].set_ylabel("3y z-score")\n'
            "plt.tight_layout(); plt.show()",
        ),
        (
            "md",
            "## Forward returns by positioning bucket\n\n"
            "Buckets are the quantiles specified in advance: bottom 10%, 10-25%, 25-75%, 75-90%, top "
            "10%. Breakpoints are cut on the full sample, so this is an **in-sample descriptive "
            "study**, not a tradable rule. Forward windows overlap heavily, so t-statistics use "
            "Newey-West standard errors; naive errors would be far too small.",
        ),
        (
            "code",
            'buckets = results["positioning_buckets"]\n'
            "order = [\"bottom 10%\", \"10-25%\", \"25-75%\", \"75-90%\", \"top 10%\"]\n"
            'view = buckets[buckets["horizon_days"] == 21].pivot_table(index="bucket", columns="market", values="mean_return_pct").reindex(order)\n'
            "view",
        ),
        (
            "code",
            "fig, axes = plt.subplots(1, 5, figsize=(15, 4), sharey=True)\n"
            'for ax, market in zip(axes, buckets["market"].unique()):\n'
            '    rows = buckets[(buckets["market"] == market) & (buckets["horizon_days"] == 21)].set_index("bucket").reindex(order)\n'
            '    colours = ["#c00000" if abs(t) > 1.96 else "#9db6cc" for t in rows["t_stat_hac"].fillna(0)]\n'
            '    ax.bar(range(len(order)), rows["mean_return_pct"], color=colours)\n'
            '    ax.axhline(0, color="#333333")\n'
            "    ax.set_xticks(range(len(order)), order, rotation=60, ha=\"right\", fontsize=7)\n"
            "    ax.set_title(market)\n"
            'axes[0].set_ylabel("mean forward 21d return (%)")\n'
            'plt.suptitle("Red bars are HAC t-statistics beyond +/-1.96")\n'
            "plt.tight_layout(); plt.show()",
        ),
        (
            "md",
            "## The formal test\n\n"
            "Top decile minus bottom decile forward return, per market and horizon.",
        ),
        ("code", 'results["positioning_extremes"]'),
        (
            "code",
            'extremes = results["positioning_extremes"]\n'
            'print(f"significant at 5%: {int(extremes[\'significant_5pct\'].sum())} of {len(extremes)} tests")\n'
            'print(f"pooled top-decile observations: {extremes["nobs_top"].sum():,}")',
        ),
        (
            "md",
            "**Conclusion.** Crowded positioning carried no reliable information about subsequent "
            "returns in this sample: neither continuation nor reversal. That was one of the three "
            "outcomes the hypothesis allowed for, and it is reported as the result rather than being "
            "replaced by a search for a specification that worked.",
        ),
    ],
    "05_backtests.ipynb": [
        (
            "md",
            "# 5. Backtests\n\n"
            "Signals, timing, costs, regimes, and the consistency test that decides whether any of "
            "it is real.\n\n" + PROVENANCE_WARNING,
        ),
        ("code", PREAMBLE),
        (
            "md",
            "## The signal registry\n\n"
            "Every candidate signal is declared, so *how many things did you test?* has an exact "
            "answer and the multiple-testing adjustment can cover the whole family rather than the "
            "survivors.",
        ),
        ("code", 'results["signal_registry"]'),
        (
            "md",
            "## The timing rule\n\n"
            "One rule in one place: the position decided at $t$ is multiplied by the return from $t$ "
            "to $t+1$. The check below is empirical rather than asserted: a signal built from "
            "today's return sign still makes nothing, because the backtester shifts it.",
        ),
        (
            "code",
            "index = pd.date_range(\"2020-01-01\", periods=500, freq=\"B\")\n"
            "returns = pd.Series(np.random.default_rng(0).normal(0, 0.01, 500), index=index)\n"
            "clairvoyant = np.sign(returns)\n"
            'lagged = backtest.run("shifted", clairvoyant, returns).returns.sum()\n'
            "leaked = (clairvoyant * returns).sum()\n"
            'print(f"same-day (leaked) cumulative return : {leaked:>8.3f}")\n'
            'print(f"as the backtester runs it           : {lagged:>8.3f}")',
        ),
        (
            "md",
            "## Headline results, in sample and out of sample side by side\n\n"
            "Costs are charged over a grid; the headline figure uses the middle of it. Financing at "
            "the effective fed funds rate applies to outright cash positions only, because hedged "
            "spreads and futures-proxy returns are already excess returns.",
        ),
        (
            "code",
            'headline = results["backtest_headline"]\n'
            'columns = ["signal", "split", "nobs", "annual_return", "annual_vol", "sharpe", "sortino",\n'
            '           "max_drawdown", "hit_rate", "profit_factor", "trades", "avg_holding_days",\n'
            '           "turnover_annual", "beta_to_benchmark"]\n'
            'headline.sort_values(["signal", "split"])[columns]',
        ),
        (
            "md",
            "## The consistency test\n\n"
            "This is the table to read before any other.",
        ),
        (
            "code",
            'consistency = results["backtest_consistency"]\n'
            "consistency",
        ),
        (
            "code",
            'survivors = int(consistency["positive_in_all_splits"].sum())\n'
            'print(f"{survivors} of {len(consistency)} strategies had a positive Sharpe in all three splits")\n'
            'best = headline[headline["split"] == "test"].sort_values("sharpe", ascending=False).iloc[0]\n'
            'print(f"\\nbest out-of-sample Sharpe: {best[\'sharpe\']:.2f} ({best[\'signal\']}, {int(best[\'trades\'])} trades)")\n'
            'train = headline[(headline["signal"] == best["signal"]) & (headline["split"] == "train")].iloc[0]\n'
            'print(f"the same strategy in training: {train[\'sharpe\']:.2f}")',
        ),
        (
            "md",
            "A strategy that loses money in training and makes it in the test window has the ordering "
            "the wrong way round for an edge and the right way round for noise.",
        ),
        (
            "md",
            "## Where the H1 strategy's return actually comes from\n\n"
            "`h1_equity_forecast` posts a respectable out-of-sample Sharpe, which appears to "
            "contradict the finding that the model has no predictive R-squared. It does not: the "
            "fitted intercept is positive, so the forecast is positive most of the time.",
        ),
        (
            "code",
            'h1 = headline[headline["signal"] == "h1_equity_forecast"]\n'
            'h1[["split", "sharpe", "share_long", "share_short", "beta_to_benchmark", "corr_to_benchmark"]]',
        ),
        (
            "md",
            "It is a low-beta long-equity position wearing a forecasting model's clothes. Reporting "
            "beta and long/short share next to Sharpe is what makes that visible.",
        ),
        ("md", "## Equity curves"),
        (
            "code",
            "fig, ax = plt.subplots(figsize=(13, 6))\n"
            'for name, result in results["backtest_results"].items():\n'
            "    ax.plot(result.returns.index, result.returns.cumsum() * 100, linewidth=1.0, label=name)\n"
            'colours = {"train": "#f2f2f2", "validation": "#fff4e0", "test": "#e8f2ff"}\n'
            "for label, (lo, hi) in config.SPLITS.items():\n"
            "    ax.axvspan(lo, hi, color=colours[label], zorder=0)\n"
            'ax.axhline(0, color="#333333"); ax.grid(alpha=0.3)\n'
            'ax.set_ylabel("cumulative log return (%)")\n'
            'ax.set_title(f"All backtested strategies at {config.HEADLINE_COST_BPS:.0f}bp")\n'
            'ax.legend(fontsize=7, ncol=2, loc="lower left")\n'
            "plt.tight_layout(); plt.show()",
        ),
        (
            "md",
            "## Transaction cost sensitivity\n\n"
            "The question is not what costs are, but whether the conclusion survives them.",
        ),
        (
            "code",
            'summary = results["backtest_summary"]\n'
            'summary[summary["split"] == "test"].pivot_table(index="signal", columns="cost_bps", values="sharpe")',
        ),
        (
            "md",
            "Turnover here is low, so the grid does not overturn any conclusion. That is a property "
            "of a multi-week average holding period, not a claim that costs do not matter: a signal "
            "trading ten times more often would be destroyed by the same grid.",
        ),
        ("md", "## Regime attribution"),
        (
            "code",
            'regime = results["regime_performance"]\n'
            'primary = regime[regime["signal"] == results["primary_strategy"]]\n'
            'primary[["dimension", "regime", "days", "annual_return", "sharpe", "max_drawdown", "hit_rate", "trades"]]',
        ),
        (
            "md",
            "Regime labels are cut on the full sample, so this is ex-post attribution, not a tradable "
            "filter. A regime-conditioned strategy using full-sample breakpoints is a lookahead bug "
            "with a respectable name.",
        ),
        ("md", "## Multiple testing"),
        (
            "code",
            'holm = results["holm"]\n'
            'print(f"family size: {len(holm)}; significant after Holm-Bonferroni: {int(holm[\'significant_5pct\'].sum())}")\n'
            "holm.head(12)",
        ),
        (
            "md",
            "## Numbers I must know\n\n"
            "Generated from the tables above, not typed in.",
        ),
        (
            "code",
            'for row in results["interview_numbers"].itertuples():\n'
            '    print(f"{row.question}\\n    {row.answer}\\n")',
        ),
    ],
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-exec", action="store_true", help="write notebooks without executing them")
    args = parser.parse_args()

    NOTEBOOK_DIR.mkdir(exist_ok=True)
    for filename, cells in NOTEBOOKS.items():
        book = notebook(cells)
        path = NOTEBOOK_DIR / filename
        nbf.write(book, path)
        print(f"wrote {path.relative_to(ROOT)} ({len(cells)} cells)")

    if args.no_exec:
        return 0

    from nbclient import NotebookClient

    for filename in NOTEBOOKS:
        path = NOTEBOOK_DIR / filename
        book = nbf.read(path, as_version=4)
        client = NotebookClient(book, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(NOTEBOOK_DIR)}})
        client.execute()
        nbf.write(book, path)
        print(f"executed {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
