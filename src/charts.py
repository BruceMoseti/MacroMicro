"""Chart generation.

Every chart is written through :func:`finish`, which stamps the observation window and
the data provenance into the figure and records the file in a manifest. That is what lets
``validation.check_charts_dated`` assert that no chart in ``outputs/charts`` is undated,
and it makes it impossible to circulate a figure from this project without the reader
seeing which dataset produced it.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config, relative_value

plt.rcParams.update(
    {
        "figure.figsize": (11, 6),
        "figure.dpi": 130,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "legend.frameon": False,
    }
)

PALETTE = ["#1f4e79", "#c00000", "#2e7d32", "#e07b00", "#6a3d9a", "#00838f", "#8d6e63"]
_MANIFEST: list[dict] = []


def reset_manifest() -> None:
    _MANIFEST.clear()


def manifest() -> pd.DataFrame:
    return pd.DataFrame(_MANIFEST)


def finish(fig, filename: str, index: pd.Index, source: str, note: str = "") -> None:
    """Stamp a figure with its date range and provenance, then save it."""
    start, end = pd.Timestamp(index.min()), pd.Timestamp(index.max())
    stamp = f"Observations {start:%Y-%m-%d} to {end:%Y-%m-%d}  |  {len(index):,} rows  |  data source: {source}"
    if note:
        stamp += f"  |  {note}"
    fig.text(0.01, 0.005, stamp, fontsize=7.0, color="#555555", ha="left")
    if source == config.SOURCE_SYNTHETIC:
        fig.text(
            0.5,
            0.5,
            "SYNTHETIC DATA",
            fontsize=44,
            color="#c00000",
            alpha=0.10,
            ha="center",
            va="center",
            rotation=28,
            zorder=10,
            transform=fig.transFigure,
        )
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    path = config.CHART_DIR / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    _MANIFEST.append(
        {
            "chart": filename,
            "start": start.date().isoformat(),
            "end": end.date().isoformat(),
            "rows": int(len(index)),
            "source": source,
            "dated": True,
        }
    )


def _date_axis(ax) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def _shade_splits(ax) -> None:
    colours = {"train": "#f2f2f2", "validation": "#fff4e0", "test": "#e8f2ff"}
    for label, (lo, hi) in config.SPLITS.items():
        ax.axvspan(lo, hi, color=colours[label], zorder=0)
    ymin, ymax = ax.get_ylim()
    for label, (lo, _) in config.SPLITS.items():
        ax.text(lo + pd.Timedelta(days=25), ymax, f" {label}", fontsize=7, color="#666666", va="top")


# --------------------------------------------------------------------------------------
# Market overview
# --------------------------------------------------------------------------------------
def cross_asset_overview(panel: pd.DataFrame, source: str) -> None:
    series = [
        ("spx", "S&P 500 (index)"),
        ("dgs10", "10y Treasury yield (%)"),
        ("usd", "Broad USD index"),
        ("wti", "WTI crude ($/bbl)"),
        ("gold", "Gold ($/oz)"),
        ("vix", "VIX"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(12, 8.5), sharex=True)
    for ax, (column, label), colour in zip(axes.ravel(), series, PALETTE):
        ax.plot(panel.index, panel[column], color=colour, linewidth=1.0)
        ax.set_title(label)
        _date_axis(ax)
    fig.suptitle("Cross-asset market overview", fontsize=13, fontweight="bold")
    finish(fig, "01_cross_asset_overview.png", panel.index, source)


def yield_curve(panel: pd.DataFrame, curve_frame: pd.DataFrame, source: str) -> None:
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, height_ratios=[1.1, 1])
    for column, label, colour in (
        ("dgs2", "2y", PALETTE[0]),
        ("dgs10", "10y", PALETTE[1]),
        ("dgs30", "30y", PALETTE[2]),
    ):
        top.plot(panel.index, panel[column], label=label, color=colour, linewidth=1.0)
    top.set_ylabel("yield (%)")
    top.legend(ncol=3, loc="upper left")
    top.set_title("Treasury yields")

    curve = curve_frame["curve_2s10s"]
    bottom.plot(curve.index, curve, color=PALETTE[0], linewidth=1.0, label="2s10s")
    bottom.plot(curve_frame.index, curve_frame["curve_2s30s"], color=PALETTE[4], linewidth=0.9, label="2s30s")
    bottom.axhline(0.0, color="#c00000", linewidth=0.8)
    bottom.fill_between(curve.index, curve, 0.0, where=curve < 0, color="#c00000", alpha=0.18, label="inverted")
    latest = curve.iloc[-1]
    bottom.annotate(
        f"{curve.index[-1]:%Y-%m-%d}: {latest:+.0f}bp",
        xy=(curve.index[-1], latest),
        xytext=(-135, -55),
        textcoords="offset points",
        fontsize=8,
        arrowprops={"arrowstyle": "->", "color": "#333333", "linewidth": 0.7},
    )
    bottom.set_ylabel("spread (bp)")
    bottom.legend(ncol=3, loc="upper left")
    bottom.set_title("Yield curve slope")
    _date_axis(bottom)
    finish(fig, "02_yield_curve.png", panel.index, source)


def real_yield_decomposition(panel: pd.DataFrame, source: str) -> None:
    fig, ax = plt.subplots()
    ax.plot(panel.index, panel["dgs10"], color="#333333", linewidth=1.1, label="10y nominal")
    ax.plot(panel.index, panel["real10"], color=PALETTE[0], linewidth=1.0, label="10y TIPS real")
    ax.plot(panel.index, panel["breakeven10"], color=PALETTE[3], linewidth=1.0, label="10y breakeven")
    ax.axhline(0.0, color="#999999", linewidth=0.6)
    ax.set_ylabel("percent")
    ax.set_title("Fisher decomposition of the 10y nominal yield: nominal = real + breakeven")
    ax.legend(ncol=3, loc="upper left")
    _date_axis(ax)
    finish(fig, "03_real_yield_decomposition.png", panel.index, source, "nominal = real + breakeven by identity")


def correlation_matrix(feature_frame: pd.DataFrame, source: str) -> None:
    columns = {
        "ret_1d_spx": "S&P 500",
        "ret_1d_ndx": "Nasdaq 100",
        "ret_1d_vix": "VIX",
        "dy_1d_dgs2": "2y yield",
        "dy_1d_dgs10": "10y yield",
        "dy_1d_real10": "10y real",
        "dy_1d_breakeven10": "breakeven",
        "dy_1d_slope": "2s10s slope",
        "ret_1d_usd": "broad USD",
        "ret_1d_wti": "WTI",
        "ret_1d_gold": "gold",
        "dy_1d_hy_oas": "HY OAS",
    }
    data = feature_frame[list(columns)].dropna().rename(columns=columns)
    matrix = data.corr()
    fig, ax = plt.subplots(figsize=(8.5, 7))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(matrix)), matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix)), matrix.columns)
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            value = matrix.iloc[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if abs(value) > 0.55 else "#222222",
            )
    fig.colorbar(image, ax=ax, shrink=0.8, label="correlation")
    ax.set_title("Daily cross-asset correlation matrix (returns and basis point changes)")
    ax.grid(False)
    finish(fig, "04_correlation_matrix.png", data.index, source)


# --------------------------------------------------------------------------------------
# Cross-asset relationships
# --------------------------------------------------------------------------------------
def rolling_correlations(rolling: pd.DataFrame, source: str) -> None:
    panels = [
        ("equity_vs_rates", "S&P 500 return vs 10y yield change"),
        ("equity_vs_vix", "S&P 500 return vs VIX change"),
        ("usd_vs_wti", "Broad USD vs WTI return"),
        ("gold_vs_real", "Gold return vs 10y real yield change"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True)
    for ax, (key, title) in zip(axes.ravel(), panels):
        for window, colour, width in ((63, PALETTE[3], 0.7), (252, PALETTE[0], 1.2)):
            column = f"corr{window}_{key}"
            if column in rolling:
                ax.plot(rolling.index, rolling[column], color=colour, linewidth=width, label=f"{window}d")
        ax.axhline(0.0, color="#c00000", linewidth=0.8)
        ax.set_ylim(-1, 1)
        ax.set_title(title)
        ax.legend(ncol=2, loc="lower left")
        _date_axis(ax)
    fig.suptitle("Rolling cross-asset correlations", fontsize=13, fontweight="bold")
    finish(fig, "05_rolling_correlations.png", rolling.index, source)


def rolling_equity_betas(rolling_beta_frame: pd.DataFrame, source: str) -> None:
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    top.plot(
        rolling_beta_frame.index,
        rolling_beta_frame["beta_dy_1d_real10"] * 100.0,
        color=PALETTE[0],
        linewidth=1.1,
        label="10y real yield",
    )
    top.plot(
        rolling_beta_frame.index,
        rolling_beta_frame["beta_dy_1d_breakeven10"] * 100.0,
        color=PALETTE[3],
        linewidth=1.0,
        label="10y breakeven",
    )
    top.axhline(0.0, color="#c00000", linewidth=0.9)
    top.set_ylabel("% equity return\nper 1bp")
    top.set_title("Rolling 252-day S&P 500 sensitivity to rate shocks (sign flips are the story)")
    top.legend(ncol=2, loc="upper left")

    bottom.plot(rolling_beta_frame.index, rolling_beta_frame["beta_ret_1d_vix"], color=PALETTE[1], linewidth=1.1)
    bottom.axhline(0.0, color="#999999", linewidth=0.6)
    bottom.set_ylabel("beta to VIX\nlog change")
    bottom.set_title("Rolling 252-day S&P 500 sensitivity to VIX (stable and negative throughout)")
    _date_axis(bottom)
    finish(fig, "06_rolling_equity_betas.png", rolling_beta_frame.index, source)


def levels_versus_changes(panel: pd.DataFrame, relationships: pd.DataFrame, source: str) -> None:
    """The spurious levels regression, shown next to the differenced version."""
    row = relationships[relationships["pair"] == relative_value.PRIMARY_PAIR].iloc[0]
    log_gold, real = np.log(panel["gold"]), panel["real10"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5.2))

    scatter = left.scatter(real, log_gold, c=mdates.date2num(panel.index), cmap="viridis", s=4, alpha=0.7)
    grid = np.linspace(real.min(), real.max(), 50)
    left.plot(grid, row["alpha"] + row["beta"] * grid, color="#c00000", linewidth=1.6)
    left.set_xlabel("10y TIPS real yield (%)")
    left.set_ylabel("log gold")
    left.set_title(f"Levels: beta = {row['beta']:+.4f} (t = {row['beta_t']:.1f}), R2 = {row['r_squared']:.3f}")
    bar = fig.colorbar(scatter, ax=left, shrink=0.85)
    bar.ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    changes = pd.concat({"dy": log_gold.diff(), "dx": real.diff()}, axis=1).dropna()
    right.scatter(changes["dx"] * 100.0, changes["dy"] * 100.0, s=4, alpha=0.35, color=PALETTE[0])
    grid = np.linspace(changes["dx"].min(), changes["dx"].max(), 50)
    right.plot(grid * 100.0, row["beta_changes"] * grid * 100.0, color="#c00000", linewidth=1.6)
    right.set_xlabel("daily change in 10y real yield (bp)")
    right.set_ylabel("daily gold return (%)")
    right.set_title(
        f"Changes: beta = {row['beta_changes']:+.4f} (t = {row['beta_changes_t']:.1f}), "
        f"R2 = {row['r_squared_changes']:.3f}"
    )
    fig.suptitle(
        "Gold and the 10y real yield: the levels regression has the wrong sign because both series trend",
        fontsize=12,
        fontweight="bold",
    )
    finish(fig, "07_levels_versus_changes.png", panel.index, source, "colour encodes calendar time")


# --------------------------------------------------------------------------------------
# Relative value
# --------------------------------------------------------------------------------------
def residual_zscore(zscores: pd.DataFrame, residuals: pd.DataFrame, pair: str, source: str) -> None:
    from . import signals

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    residual = residuals[pair].dropna()
    top.plot(residual.index, residual, color=PALETTE[0], linewidth=0.9)
    top.axhline(0.0, color="#999999", linewidth=0.6)
    top.set_ylabel("residual")
    top.set_title(f"{pair}: rolling-beta residual (hedge ratio estimated on a trailing 252-day window)")

    zscore = zscores[pair].dropna()
    bottom.plot(zscore.index, zscore, color="#333333", linewidth=0.8)
    for level, style in ((signals.ENTRY_Z, "-"), (-signals.ENTRY_Z, "-")):
        bottom.axhline(level, color="#c00000", linewidth=0.9, linestyle=style)
    for level in (signals.EXIT_Z, -signals.EXIT_Z):
        bottom.axhline(level, color="#2e7d32", linewidth=0.8, linestyle="--")
    bottom.fill_between(zscore.index, zscore, signals.ENTRY_Z, where=zscore > signals.ENTRY_Z, color="#c00000", alpha=0.25)
    bottom.fill_between(
        zscore.index, zscore, -signals.ENTRY_Z, where=zscore < -signals.ENTRY_Z, color="#2e7d32", alpha=0.25
    )
    bottom.set_ylabel("z-score (60d)")
    bottom.set_title(
        f"Entry at |z| > {signals.ENTRY_Z} (red), exit at |z| < {signals.EXIT_Z} (green dashed); "
        f"{int((zscore.abs() > signals.ENTRY_Z).sum())} days beyond the entry band"
    )
    _date_axis(bottom)
    finish(fig, "08_rv_residual_zscore.png", zscore.index, source)


def rolling_hedge_ratios(betas: pd.DataFrame, source: str) -> None:
    fig, axes = plt.subplots(len(betas.columns), 1, figsize=(11, 2.0 * len(betas.columns)), sharex=True)
    for ax, column, colour in zip(np.atleast_1d(axes), betas.columns, PALETTE):
        series = betas[column].dropna()
        ax.plot(series.index, series, color=colour, linewidth=1.0)
        ax.axhline(0.0, color="#c00000", linewidth=0.8)
        ax.set_ylabel(column, fontsize=7.5)
    np.atleast_1d(axes)[0].set_title(
        "Rolling 252-day hedge ratios. Instability here is why some residuals are not tradable"
    )
    _date_axis(np.atleast_1d(axes)[-1])
    finish(fig, "09_rolling_hedge_ratios.png", betas.index, source)


# --------------------------------------------------------------------------------------
# Positioning
# --------------------------------------------------------------------------------------
def positioning_history(feature_frame: pd.DataFrame, market: str, label: str, source: str) -> None:
    ratio = feature_frame[f"cot_{market}_ratio"].dropna()
    zscore = feature_frame[f"cot_{market}_z156"].dropna()
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    top.plot(ratio.index, ratio * 100.0, color=PALETTE[0], linewidth=1.0)
    top.axhline(0.0, color="#999999", linewidth=0.6)
    top.set_ylabel("net position\n(% of open interest)")
    top.set_title(f"{label}: speculative net positioning, aligned to its publication session")

    bottom.plot(zscore.index, zscore, color=PALETTE[1], linewidth=1.0)
    for level in (2, -2):
        bottom.axhline(level, color="#c00000", linewidth=0.8, linestyle="--")
    bottom.axhline(0.0, color="#999999", linewidth=0.6)
    bottom.set_ylabel("3y z-score")
    bottom.set_title("Three-year (156 week) trailing z-score of the position ratio")
    _date_axis(bottom)
    finish(fig, "10_positioning_history.png", ratio.index, source, "Tuesday positions, usable from the following Monday")


def positioning_buckets(bucket_study: pd.DataFrame, horizon: int, source: str, index: pd.Index) -> None:
    subset = bucket_study[bucket_study["horizon_days"] == horizon]
    order = ["bottom 10%", "10-25%", "25-75%", "75-90%", "top 10%"]
    markets = subset["market"].unique()
    fig, axes = plt.subplots(1, len(markets), figsize=(2.7 * len(markets), 4.4), sharey=True)
    for ax, market in zip(np.atleast_1d(axes), markets):
        rows = subset[subset["market"] == market].set_index("bucket").reindex(order)
        colours = ["#c00000" if abs(t) > 1.96 else "#9db6cc" for t in rows["t_stat_hac"].fillna(0.0)]
        ax.bar(range(len(order)), rows["mean_return_pct"], color=colours)
        ax.axhline(0.0, color="#333333", linewidth=0.8)
        ax.set_xticks(range(len(order)), order, rotation=55, ha="right", fontsize=7)
        ax.set_title(rows["label"].dropna().iloc[0] if rows["label"].notna().any() else market, fontsize=9)
        ax.grid(axis="x", visible=False)
    np.atleast_1d(axes)[0].set_ylabel(f"mean forward {horizon}d return (%)")
    fig.suptitle(
        f"Forward {horizon}-day return by positioning bucket (red = HAC t-statistic beyond +/-1.96)",
        fontsize=12,
        fontweight="bold",
    )
    finish(fig, f"11_positioning_buckets_{horizon}d.png", index, source, "in-sample descriptive study")


# --------------------------------------------------------------------------------------
# Backtests
# --------------------------------------------------------------------------------------
def equity_curves(returns: dict[str, pd.Series], source: str, filename: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6))
    index = None
    for (name, series), colour in zip(returns.items(), PALETTE * 3):
        curve = series.dropna().cumsum() * 100.0
        ax.plot(curve.index, curve, label=name, color=colour, linewidth=1.1)
        index = curve.index if index is None else index.union(curve.index)
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    _shade_splits(ax)
    ax.set_ylabel("cumulative log return (%)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    _date_axis(ax)
    finish(fig, filename, index, source, f"costs {config.HEADLINE_COST_BPS:.0f}bp per unit traded")


def cost_sensitivity(summary: pd.DataFrame, source: str, index: pd.Index) -> None:
    subset = summary[summary["split"] == "test"]
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    for (name, group), colour in zip(subset.groupby("signal"), PALETTE * 3):
        ordered = group.sort_values("cost_bps")
        ax.plot(ordered["cost_bps"], ordered["sharpe"], marker="o", markersize=4, label=name, color=colour, linewidth=1.1)
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_xlabel("transaction cost (bp per unit traded)")
    ax.set_ylabel("out-of-sample Sharpe")
    ax.set_title("Transaction cost sensitivity, true out-of-sample period (2025-01-01 to 2026-08-13)")
    ax.legend(fontsize=7.5, ncol=2)
    finish(fig, "13_cost_sensitivity.png", index, source, "cost assumption is a sensitivity, not a sourced model")


def regime_bars(regime_table: pd.DataFrame, signal: str, source: str, index: pd.Index) -> None:
    subset = regime_table[regime_table["signal"] == signal]
    dimensions = subset["dimension"].unique()
    fig, axes = plt.subplots(1, len(dimensions), figsize=(3.1 * len(dimensions), 4.6), sharey=True)
    for ax, dimension in zip(np.atleast_1d(axes), dimensions):
        rows = subset[subset["dimension"] == dimension]
        colours = ["#2e7d32" if v > 0 else "#c00000" for v in rows["sharpe"]]
        ax.bar(range(len(rows)), rows["sharpe"], color=colours)
        ax.axhline(0.0, color="#333333", linewidth=0.8)
        labels = [f"{r.regime}\n({r.days}d, {int(r.trades)} trades)" for r in rows.itertuples()]
        ax.set_xticks(range(len(rows)), labels, rotation=55, ha="right", fontsize=6.5)
        ax.set_title(dimension, fontsize=9)
        ax.grid(axis="x", visible=False)
    np.atleast_1d(axes)[0].set_ylabel("Sharpe within regime")
    fig.suptitle(f"{signal}: performance by regime (ex-post attribution, full sample)", fontsize=12, fontweight="bold")
    finish(fig, "14_regime_performance.png", index, source)


def rolling_performance(rolling: pd.DataFrame, signal: str, source: str) -> None:
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 6.8), sharex=True, height_ratios=[1.2, 1])
    top.plot(rolling.index, rolling["rolling_sharpe_252"], color=PALETTE[0], linewidth=1.0)
    top.axhline(0.0, color="#c00000", linewidth=0.8)
    top.set_ylabel("rolling 252d Sharpe")
    top.set_title(f"{signal}: rolling Sharpe and drawdown")
    bottom.fill_between(rolling.index, rolling["drawdown"] * 100.0, 0.0, color="#c00000", alpha=0.35)
    bottom.set_ylabel("drawdown (%)")
    _date_axis(bottom)
    finish(fig, "15_rolling_performance.png", rolling.index, source)
