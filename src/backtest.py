"""Backtester.

One timing rule, applied everywhere:

    strategy return at t+1 = position decided at t  x  asset return from t to t+1
                             - |position(t) - position(t-1)| x cost

The position series is shifted forward one session before it meets a return, so a signal
can never earn the move that generated it. Costs are charged on the change in position,
on the day the position changes.

Transaction costs are run over the grid in ``config.COST_GRID_BPS`` and reported as a
sensitivity. They are not a sourced cost model for any specific instrument, and the
report says so.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config, metrics


VOL_TARGET = 0.10
VOL_LOOKBACK = 63
MAX_LEVERAGE = 3.0


def size_positions(
    states: pd.Series,
    traded_returns: pd.Series,
    target_vol: float = VOL_TARGET,
    lookback: int = VOL_LOOKBACK,
    max_leverage: float = MAX_LEVERAGE,
) -> pd.Series:
    """Scale a -1/0/+1 signal to a constant ex-ante volatility target.

    Without this, reported volatility and Sharpe are driven by how volatile each traded
    leg happens to be rather than by signal quality, and a pair whose rolling hedge ratio
    is unstable can dominate the whole comparison. The scale factor is
    ``target_vol / trailing annualised volatility``, computed from a trailing window,
    lagged one day, and capped at ``max_leverage``.

    The size is fixed when the position is opened and held for the life of the trade.
    Rescaling daily would make the position change every session, so turnover and trade
    counts would no longer mean anything. A trade opened before the volatility window is
    populated has no size and is therefore not taken; every signal in this project has a
    longer warm-up than the volatility window, so that only affects the leading rows.
    """
    trailing_vol = traded_returns.rolling(lookback, min_periods=lookback // 2).std() * np.sqrt(
        config.TRADING_DAYS_PER_YEAR
    )
    scale = (target_vol / trailing_vol.shift(1)).clip(upper=max_leverage)

    state_changed = states.ne(states.shift(1))
    entry_scale = scale.where(state_changed).ffill()
    return (states * entry_scale).fillna(0.0)


@dataclass
class BacktestResult:
    name: str
    returns: pd.Series
    positions: pd.Series
    trades: pd.Series
    cost_bps: float
    gross_returns: pd.Series
    costs: pd.Series

    def stats(self, split: str = "full", benchmark: pd.Series | None = None) -> dict:
        """Statistics for one split. Returns are already excess, so no further cash
        adjustment is applied here."""
        returns = self.returns if split == "full" else config.slice_split(self.returns, split)
        positions = self.positions if split == "full" else config.slice_split(self.positions, split)
        trades = self.trades if split == "full" else config.slice_split(self.trades, split)
        row = {"signal": self.name, "split": split, "cost_bps": self.cost_bps}
        row.update(metrics.performance(returns, positions=positions, trades=trades, benchmark=benchmark))
        if len(positions):
            held = positions.shift(1).fillna(0.0)
            row["share_long"] = float((held > 0).mean())
            row["share_short"] = float((held < 0).mean())
        if len(returns):
            row["start"] = returns.index.min().date().isoformat()
            row["end"] = returns.index.max().date().isoformat()
        return row


def run(
    name: str,
    positions: pd.Series,
    asset_returns: pd.Series,
    cost_bps: float = 0.0,
    financing_pct: pd.Series | None = None,
) -> BacktestResult:
    """Apply a position series to an asset return series with linear costs.

    Every strategy return in this project is an excess return over cash. Pass
    ``financing_pct`` (an annualised percentage rate) for an outright position in a cash
    asset, where a long is funded and a short earns the rate; the asset return is
    converted to an excess return once, before positions are applied, so no financing is
    charged on days the strategy is flat. Leave it as None for beta-hedged spreads and
    futures-proxy returns, which are already excess returns and would otherwise be
    charged financing twice.
    """
    aligned = pd.concat({"position": positions, "asset": asset_returns}, axis=1).dropna(subset=["asset"])
    position = aligned["position"].fillna(0.0)
    asset = aligned["asset"]
    if financing_pct is not None:
        daily_rate = financing_pct.reindex(asset.index).ffill() / 100.0 / config.TRADING_DAYS_PER_YEAR
        asset = asset - daily_rate

    # The single line that enforces the timing rule.
    lagged = position.shift(1).fillna(0.0)
    gross = lagged * asset

    position_change = lagged.diff().abs().fillna(lagged.abs())
    costs = position_change * cost_bps / 10_000.0
    trades = (position_change > 1e-12).astype(int)

    net = gross - costs
    return BacktestResult(
        name=name,
        returns=net.rename(name),
        positions=position.rename(name),
        trades=trades.rename(name),
        cost_bps=cost_bps,
        gross_returns=gross.rename(name),
        costs=costs.rename(name),
    )


def cost_sensitivity(
    name: str,
    positions: pd.Series,
    asset_returns: pd.Series,
    financing_pct: pd.Series | None = None,
    grid: tuple[float, ...] = config.COST_GRID_BPS,
    benchmark: pd.Series | None = None,
) -> pd.DataFrame:
    """Headline statistics for one signal across the cost grid and every split."""
    rows = []
    for cost in grid:
        result = run(name, positions, asset_returns, cost_bps=cost, financing_pct=financing_pct)
        for split in ("full", "train", "validation", "test"):
            benchmark_slice = None if benchmark is None else (
                benchmark if split == "full" else config.slice_split(benchmark, split)
            )
            stats = result.stats(split, benchmark=benchmark_slice)
            if stats.get("nobs"):
                rows.append(stats)
    return pd.DataFrame(rows)


def split_consistency(headline: pd.DataFrame) -> pd.DataFrame:
    """Sharpe in each split side by side, and whether the sign holds across all three.

    This is the table that decides whether anything here is real. A strategy that loses
    money in training and makes it in the test window has the ordering the wrong way
    round for an edge, and reporting only its out-of-sample number would be misleading.
    """
    wide = headline.pivot_table(index="signal", columns="split", values="sharpe")
    for split in ("train", "validation", "test"):
        if split not in wide:
            wide[split] = np.nan
    wide = wide[["train", "validation", "test"] + (["full"] if "full" in wide else [])]
    wide["positive_in_all_splits"] = (wide[["train", "validation", "test"]] > 0).all(axis=1)
    wide["splits_positive"] = (wide[["train", "validation", "test"]] > 0).sum(axis=1)
    return wide.reset_index()


def regime_performance(result: BacktestResult, regimes: pd.DataFrame) -> pd.DataFrame:
    """Strategy statistics inside each regime bucket.

    Regime labels are assigned from the whole sample, so this is ex-post attribution and
    not a tradable filter. Using it as a filter would require thresholds estimated from
    trailing data only.
    """
    rows = []
    for dimension in regimes.columns:
        labels = regimes[dimension].dropna()
        for bucket in labels.unique():
            index = labels[labels == bucket].index.intersection(result.returns.index)
            if len(index) < 60:
                continue
            stats = metrics.performance(
                result.returns.loc[index],
                positions=result.positions.loc[index],
                trades=result.trades.loc[index],
            )
            rows.append(
                {
                    "signal": result.name,
                    "cost_bps": result.cost_bps,
                    "dimension": dimension,
                    "regime": bucket,
                    "days": len(index),
                    "annual_return": stats["annual_return"],
                    "annual_vol": stats["annual_vol"],
                    "sharpe": stats["sharpe"],
                    "max_drawdown": stats["max_drawdown"],
                    "hit_rate": stats["hit_rate"],
                    "trades": stats.get("trades", np.nan),
                }
            )
    return pd.DataFrame(rows)
