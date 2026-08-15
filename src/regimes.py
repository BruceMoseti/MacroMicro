"""Quantitative regime classification.

Four dimensions, each with fixed rules so "market conditions" is a measurable label
rather than a description:

volatility    VIX terciles cut on the full sample.
curve         the 2s10s slope: inverted, 0-100bp, or above 100bp. Fixed economic
              breakpoints, not quantiles.
real_yield    10y TIPS real yield: bottom quartile, middle 50%, top quartile.
usd           broad dollar direction, from the sign of the trailing 63 day return.

The volatility, real-yield and curve-quantile breakpoints use the whole sample, so these
labels are for ex-post attribution only. That is a deliberate choice: the question being
answered is "where did this strategy make and lose money", which is a description of
history. A regime *filter* inside a strategy would have to use trailing breakpoints, and
the report says so rather than leaving the reader to wonder.

The USD dimension is the exception: a trailing 63 day return is knowable in real time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def classify(panel: pd.DataFrame, curve_frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=panel.index)

    vix_low, vix_high = panel["vix"].quantile([1 / 3, 2 / 3])
    out["volatility"] = pd.cut(
        panel["vix"],
        bins=[-np.inf, vix_low, vix_high, np.inf],
        labels=[f"low VIX (<{vix_low:.1f})", f"mid VIX ({vix_low:.1f}-{vix_high:.1f})", f"high VIX (>{vix_high:.1f})"],
    )

    curve = curve_frame["curve_2s10s"]
    out["curve"] = pd.cut(
        curve,
        bins=[-np.inf, 0.0, 100.0, np.inf],
        labels=["inverted", "0-100bp", ">100bp"],
    )

    real_low, real_high = panel["real10"].quantile([0.25, 0.75])
    out["real_yield"] = pd.cut(
        panel["real10"],
        bins=[-np.inf, real_low, real_high, np.inf],
        labels=[
            f"bottom quartile (<{real_low:.2f}%)",
            f"middle 50% ({real_low:.2f}-{real_high:.2f}%)",
            f"top quartile (>{real_high:.2f}%)",
        ],
    )

    dollar_trend = np.log(panel["usd"]).diff(63)
    out["usd"] = pd.Series(
        np.where(dollar_trend > 0, "USD strengthening", "USD weakening"), index=panel.index
    ).where(dollar_trend.notna())

    return out


def summary(regime_frame: pd.DataFrame) -> pd.DataFrame:
    """Day counts per bucket, so every regime statistic has a visible sample size."""
    rows = []
    for dimension in regime_frame.columns:
        counts = regime_frame[dimension].value_counts(dropna=True)
        for bucket, days in counts.items():
            rows.append(
                {
                    "dimension": dimension,
                    "regime": bucket,
                    "days": int(days),
                    "share": float(days / regime_frame[dimension].notna().sum()),
                }
            )
    return pd.DataFrame(rows)
