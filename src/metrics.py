"""Performance statistics.

All return inputs are daily log returns unless stated. Annualisation uses
``config.TRADING_DAYS_PER_YEAR``. Sharpe ratios are computed against the daily effective
fed funds rate when one is supplied, not against zero, because over this window the cash
rate ranged from roughly zero to above 5% and treating it as zero would flatter every
result estimated during the hiking cycle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _annualisation() -> float:
    return float(config.TRADING_DAYS_PER_YEAR)


def excess_returns(returns: pd.Series, risk_free_pct: pd.Series | None) -> pd.Series:
    """Subtract the daily equivalent of an annualised percentage cash rate."""
    if risk_free_pct is None:
        return returns
    daily = risk_free_pct.reindex(returns.index).ffill() / 100.0 / _annualisation()
    return returns - daily


def max_drawdown(returns: pd.Series) -> float:
    equity = returns.cumsum()
    return float((equity - equity.cummax()).min())


def drawdown_series(returns: pd.Series) -> pd.Series:
    equity = returns.cumsum()
    return equity - equity.cummax()


def performance(
    returns: pd.Series,
    positions: pd.Series | None = None,
    trades: pd.Series | None = None,
    benchmark: pd.Series | None = None,
    risk_free_pct: pd.Series | None = None,
) -> dict:
    """The full statistics block reported for every strategy and every split."""
    returns = returns.dropna()
    n = len(returns)
    if n == 0:
        return {"nobs": 0}

    annual = _annualisation()
    excess = excess_returns(returns, risk_free_pct)
    volatility = float(returns.std() * np.sqrt(annual))
    downside = returns[returns < 0]
    downside_vol = float(downside.std() * np.sqrt(annual)) if len(downside) > 1 else np.nan
    wins, losses = returns[returns > 0], returns[returns < 0]

    # Hit rate is measured over days the strategy actually held risk. Including flat days
    # would cap it at the exposure share and make a selective strategy look wrong.
    active = returns[returns != 0] if positions is None else returns[positions.reindex(returns.index).shift(1).fillna(0.0) != 0]

    out = {
        "nobs": n,
        "years": n / annual,
        "annual_return": float(returns.mean() * annual),
        "annual_vol": volatility,
        "sharpe": float(excess.mean() * annual / volatility) if volatility > 0 else np.nan,
        "sortino": float(excess.mean() * annual / downside_vol) if downside_vol and downside_vol > 0 else np.nan,
        "max_drawdown": max_drawdown(returns),
        "calmar": float(returns.mean() * annual / abs(max_drawdown(returns))) if max_drawdown(returns) < 0 else np.nan,
        "hit_rate": float((active > 0).mean()) if len(active) else np.nan,
        "active_days": int(len(active)),
        "profit_factor": float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else np.nan,
        "avg_win": float(wins.mean()) if len(wins) else np.nan,
        "avg_loss": float(losses.mean()) if len(losses) else np.nan,
        "best_day": float(returns.max()),
        "worst_day": float(returns.min()),
        "skew": float(returns.skew()),
        "kurtosis": float(returns.kurtosis()),
    }

    monthly = returns.resample("ME").sum()
    out["best_month"] = float(monthly.max()) if len(monthly) else np.nan
    out["worst_month"] = float(monthly.min()) if len(monthly) else np.nan
    out["positive_months"] = float((monthly > 0).mean()) if len(monthly) else np.nan

    if positions is not None:
        aligned = positions.reindex(returns.index)
        out["market_exposure"] = float((aligned.abs() > 0).mean())
        out["avg_abs_position"] = float(aligned.abs().mean())
        out["turnover_annual"] = float(aligned.diff().abs().sum() / (n / annual))
    if trades is not None:
        trade_count = int(trades.reindex(returns.index).fillna(0).sum())
        out["trades"] = trade_count
        exposed_days = float((positions.reindex(returns.index).abs() > 0).sum()) if positions is not None else np.nan
        out["avg_holding_days"] = exposed_days / trade_count if trade_count else np.nan
    if benchmark is not None:
        both = pd.concat({"strategy": returns, "benchmark": benchmark.reindex(returns.index)}, axis=1).dropna()
        if len(both) > 30 and both["benchmark"].var() > 0:
            out["beta_to_benchmark"] = float(both.cov().iloc[0, 1] / both["benchmark"].var())
            out["corr_to_benchmark"] = float(both["strategy"].corr(both["benchmark"]))
    return out


def rolling_analysis(
    returns: pd.Series, benchmark: pd.Series | None = None, risk_free_pct: pd.Series | None = None
) -> pd.DataFrame:
    """Rolling Sharpe, volatility, drawdown and beta on a 252 day window."""
    annual = _annualisation()
    window = config.ROLLING_BETA_WINDOW
    excess = excess_returns(returns, risk_free_pct)
    out = pd.DataFrame(index=returns.index)
    out["rolling_vol_252"] = returns.rolling(window).std() * np.sqrt(annual)
    out["rolling_sharpe_252"] = (excess.rolling(window).mean() * annual) / out["rolling_vol_252"]
    out["rolling_return_252"] = returns.rolling(window).sum()
    out["drawdown"] = drawdown_series(returns)
    if benchmark is not None:
        aligned = benchmark.reindex(returns.index)
        covariance = returns.rolling(window).cov(aligned)
        out["rolling_beta_252"] = covariance / aligned.rolling(window).var()
    return out


def rolling_relationships(feature_frame: pd.DataFrame, pairs: dict[str, tuple[str, str]]) -> pd.DataFrame:
    """Rolling correlations at 63 and 252 days plus a 252 day rolling beta."""
    columns = {}
    for label, (left, right) in pairs.items():
        if left not in feature_frame or right not in feature_frame:
            continue
        a, b = feature_frame[left], feature_frame[right]
        for window in config.ROLLING_CORR_WINDOWS:
            columns[f"corr{window}_{label}"] = a.rolling(window, min_periods=window // 2).corr(b)
        covariance = a.rolling(config.ROLLING_BETA_WINDOW, min_periods=126).cov(b)
        variance = b.rolling(config.ROLLING_BETA_WINDOW, min_periods=126).var()
        columns[f"beta252_{label}"] = covariance / variance.replace(0.0, np.nan)
    out = pd.concat(columns, axis=1)
    out.index.name = "date"
    return out
