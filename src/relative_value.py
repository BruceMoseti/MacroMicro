"""Hypothesis 2: cross-asset relative value.

Pairs are declared here, before any performance is measured, and each one carries the
economic prior that motivated it. None was chosen by searching for the best Sharpe
ratio; the primary pair is the one with the cleanest theoretical link, not the best
backtest.

Two residuals are computed for every pair, and the distinction matters:

``static``   residual from a full-sample OLS. Used only for description and for the
             stationarity tests, because a full-sample beta is not knowable in real
             time. Anything computed from it is in-sample by construction.
``rolling``  residual from a trailing-window OLS, so the hedge ratio at date t uses only
             data up to t. This is the residual the backtest trades.

Stationarity is assessed on the training split, because that is the sample that was
available when the strategy was specified. The full-sample result is reported alongside
it for completeness, not as the decision rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, adfuller, coint

from . import config

ROLLING_BETA_WINDOW = 252
RESIDUAL_ZSCORE_WINDOW = 60

# Modified duration used to express a 10y real-yield or breakeven hedge as a TIPS
# position, matching the approximation used for the 10y note in config.ZN_MODIFIED_DURATION.
TIPS_MODIFIED_DURATION = 8.4
# Bond maths needs the yield change in decimal, but this platform stores yields in
# percent, so a duration used as a return multiplier carries the percent-to-decimal factor.
TIPS_RETURN_SCALE = TIPS_MODIFIED_DURATION / 100.0


@dataclass(frozen=True)
class Pair:
    """One relative-value relationship: y regressed on x."""

    name: str
    y: str
    x: str
    y_transform: str  # "log" for price levels, "level" for yields and spreads
    x_transform: str
    rationale: str
    tradable_leg: str  # the leg the backtest takes risk in
    tradable_note: str
    # Multiplier converting a one-unit change in the residual into a return. 1.0 when the
    # y leg is a log price. A modified duration when the y leg is a yield in percent.
    # None marks a pair whose P&L needs more than one instrument, so it is not backtested.
    return_scale: float | None


PAIRS: tuple[Pair, ...] = (
    Pair(
        "gold_real10",
        "gold",
        "real10",
        "log",
        "level",
        "Gold is a zero-coupon real asset with no cash flows, so its price should fall "
        "as the real risk-free rate rises and the opportunity cost of holding it "
        "increases. This is the cleanest theoretical link in the universe and is the "
        "pre-specified primary pair.",
        "gold",
        "Gold is traded outright. The real-yield leg would require a TIPS or inflation "
        "swap position, which is not in the cash universe, so the hedged variant expresses "
        "it as a duration-approximated TIPS position.",
        return_scale=1.0,
    ),
    Pair(
        "wti_usd",
        "wti",
        "usd",
        "log",
        "log",
        "Crude is invoiced in dollars, so a stronger dollar mechanically raises the "
        "local-currency cost for non-US buyers and weighs on the dollar price.",
        "wti",
        "The dollar leg is a trade-weighted index, not a directly tradable instrument; "
        "replicating it needs an FX basket or dollar-index futures.",
        return_scale=1.0,
    ),
    Pair(
        "breakeven_wti",
        "breakeven10",
        "wti",
        "level",
        "log",
        "Energy passes into headline inflation, and 10y breakevens historically track "
        "the oil complex even though the pass-through to a decade of inflation should "
        "be small. Extremes in the residual are candidate over-reactions.",
        "breakeven10",
        "The breakeven leg needs a TIPS-versus-nominal or inflation-swap position. A long "
        "breakeven gains when expectations rise, so the residual is scaled by a positive "
        "10y modified duration to express it as a return.",
        return_scale=TIPS_RETURN_SCALE,
    ),
    Pair(
        "spx_vix",
        "spx",
        "vix",
        "log",
        "log",
        "Implied volatility is a price of insurance on the index, so the two are "
        "mechanically linked. The residual measures whether volatility is high or low "
        "relative to where the index is trading.",
        "spx",
        "The VIX leg is not investable spot; replicating it requires variance swaps or "
        "VIX futures, which carry their own term structure.",
        return_scale=1.0,
    ),
    Pair(
        "curve_hyoas",
        "curve_2s10s",
        "hy_oas",
        "level",
        "level",
        "The curve and credit spreads both price the expected path of growth and policy. "
        "A curve that is steep relative to credit conditions is a candidate dislocation.",
        "curve_2s10s",
        "Trading the curve leg requires a duration-weighted 2s10s steepener, whose P&L is "
        "not a single-duration transform of the spread. The pair is analysed but cannot be "
        "converted into one return series, so it is not backtested.",
        return_scale=None,
    ),
)
PRIMARY_PAIR = "gold_real10"
BACKTEST_PAIRS = tuple(p.name for p in PAIRS if p.return_scale is not None)
# Pairs whose tradable leg is an outright cash asset, so the single-leg variant is
# implementable without a swap or futures overlay.
SINGLE_LEG_PAIRS = ("gold_real10", "wti_usd", "spx_vix")


def _transform(series: pd.Series, how: str) -> pd.Series:
    return np.log(series) if how == "log" else series


def pair_series(panel: pd.DataFrame, curve_frame: pd.DataFrame, pair: Pair) -> tuple[pd.Series, pd.Series]:
    """The transformed y and x legs of a pair, aligned and de-NaN'd."""
    source = pd.concat([panel, curve_frame], axis=1)
    y = _transform(source[pair.y], pair.y_transform).rename(pair.y)
    x = _transform(source[pair.x], pair.x_transform).rename(pair.x)
    aligned = pd.concat([y, x], axis=1).dropna()
    return aligned[pair.y], aligned[pair.x]


def static_residual(y: pd.Series, x: pd.Series) -> tuple[pd.Series, dict]:
    """Full-sample OLS residual. In-sample by construction; description only."""
    model = sm.OLS(y, sm.add_constant(x)).fit()
    return y - model.predict(sm.add_constant(x)), {
        "alpha": float(model.params.iloc[0]),
        "beta": float(model.params.iloc[1]),
        "beta_t": float(model.tvalues.iloc[1]),
        "r_squared": float(model.rsquared),
        "nobs": int(model.nobs),
    }


def change_relationship(y: pd.Series, x: pd.Series) -> dict:
    """The same pair estimated in first differences.

    Two trending near-unit-root series can produce a large, significant and
    economically wrong beta in levels; the differenced regression is not exposed to
    that. Reporting both makes any spurious levels result visible instead of hidden.
    """
    data = pd.concat({"dy": y.diff(), "dx": x.diff()}, axis=1).dropna()
    model = sm.OLS(data["dy"], sm.add_constant(data["dx"])).fit(
        cov_type="HAC", cov_kwds={"maxlags": 8}
    )
    return {
        "beta_changes": float(model.params.iloc[1]),
        "beta_changes_t": float(model.tvalues.iloc[1]),
        "r_squared_changes": float(model.rsquared),
        "corr_changes": float(data["dy"].corr(data["dx"])),
    }


def rolling_residual(y: pd.Series, x: pd.Series, window: int = ROLLING_BETA_WINDOW) -> pd.DataFrame:
    """Causal residual: the hedge ratio at t is estimated on (t-window, t].

    Implemented from rolling moments rather than a loop of regressions; for a single
    regressor beta = cov(x, y) / var(x) and alpha = mean(y) - beta mean(x), all from
    trailing windows.
    """
    minimum = max(60, window // 2)
    mean_y = y.rolling(window, min_periods=minimum).mean()
    mean_x = x.rolling(window, min_periods=minimum).mean()
    variance_x = x.rolling(window, min_periods=minimum).var()
    covariance = y.rolling(window, min_periods=minimum).cov(x)

    beta = covariance / variance_x.replace(0.0, np.nan)
    alpha = mean_y - beta * mean_x
    fitted = alpha + beta * x
    residual = y - fitted
    zscore = (residual - residual.rolling(RESIDUAL_ZSCORE_WINDOW, min_periods=30).mean()) / residual.rolling(
        RESIDUAL_ZSCORE_WINDOW, min_periods=30
    ).std()
    return pd.DataFrame({"alpha": alpha, "beta": beta, "fitted": fitted, "residual": residual, "zscore": zscore})


def spread_returns(y: pd.Series, x: pd.Series, beta: pd.Series) -> pd.Series:
    """Return on the beta-hedged spread: d(residual) holding the hedge ratio fixed.

    A relative-value residual is a two-legged position. Trading only the leg that
    happens to be investable leaves the position exposed to that asset's drift, which
    over a ten year window can be far larger than the convergence being harvested. The
    hedged return is

        d(residual)_t = dy_t - beta_{t-1} dx_t

    with the hedge ratio lagged so it is known before the move it hedges. Both legs are
    in the same units as y, because beta already carries the conversion.

    Where x is a yield, the hedge is a duration position: a change of dx percent in a 10y
    real yield is expressed as a TIPS return of -D x dx, so the implied instrument is a
    TIPS position rather than a cash holding. That is an approximation, and the pair's
    ``tradable_note`` records which legs need instruments outside the cash universe.
    """
    return y.diff() - beta.shift(1) * x.diff()


def convergence_study(
    residual: pd.Series,
    zscore: pd.Series,
    tradable_return: pd.Series,
    horizons: tuple[int, ...] = (5, 10, 21),
) -> pd.DataFrame:
    """Where does the residual's convergence actually show up?

    For each horizon, the mean forward change in the residual and the mean forward return
    on the tradable leg, conditioned on the residual being rich (z > 2) or cheap (z < -2),
    with the unconditional return alongside for comparison.

    This is the evidence for the project's central relative-value finding: a residual can
    converge while the tradable leg does not, in which case the position is left holding
    the asset's drift instead of the convergence.
    """
    rows = []
    for horizon in horizons:
        forward_residual = residual.shift(-horizon) - residual
        forward_return = tradable_return.shift(-1).rolling(horizon).sum().shift(-(horizon - 1))
        data = pd.concat(
            {"z": zscore, "d_residual": forward_residual, "leg_return": forward_return}, axis=1, sort=True
        ).dropna()
        rich, cheap = data[data["z"] > 2.0], data[data["z"] < -2.0]
        rows.append(
            {
                "horizon_days": horizon,
                "n_rich": int(len(rich)),
                "n_cheap": int(len(cheap)),
                "d_residual_when_rich": float(rich["d_residual"].mean()),
                "d_residual_when_cheap": float(cheap["d_residual"].mean()),
                "leg_return_when_rich_pct": float(rich["leg_return"].mean() * 100.0),
                "leg_return_when_cheap_pct": float(cheap["leg_return"].mean() * 100.0),
                "unconditional_leg_return_pct": float(
                    tradable_return.rolling(horizon).sum().mean() * 100.0
                ),
            }
        )
    return pd.DataFrame(rows)


def half_life(series: pd.Series) -> tuple[float, float]:
    """AR(1) persistence and implied half-life in trading days.

    x_t = c + phi x_{t-1} + e_t, half-life = -ln 2 / ln phi. A phi at or above 1 means
    no measurable mean reversion, reported as infinite.
    """
    data = pd.concat({"x": series, "lag": series.shift(1)}, axis=1).dropna()
    if len(data) < 30:
        return float("nan"), float("nan")
    model = sm.OLS(data["x"], sm.add_constant(data["lag"])).fit()
    phi = float(model.params.iloc[1])
    if phi <= 0 or phi >= 1:
        return phi, float("inf")
    return phi, float(-np.log(2.0) / np.log(phi))


def stationarity_report(residual: pd.Series, y: pd.Series, x: pd.Series, label: str, sample: str) -> dict:
    """ADF on the residual, Engle-Granger on the pair, AR(1) persistence and ACF.

    The plain ADF applied to a *fitted* residual is biased toward rejecting the unit
    root, because the regression already minimised that residual's variance. The
    Engle-Granger test uses critical values that account for the estimated
    cointegrating vector, so it is the one to believe.
    """
    residual = residual.dropna()
    adf_stat, adf_p, _, adf_n, adf_crit, _ = adfuller(residual.to_numpy(), autolag="AIC")
    common = y.index.intersection(x.index).intersection(residual.index)
    eg_stat, eg_p, eg_crit = coint(y.loc[common].to_numpy(), x.loc[common].to_numpy())
    phi, hl = half_life(residual)
    autocorrelation = acf(residual.to_numpy(), nlags=5, fft=False)
    return {
        "pair": label,
        "sample": sample,
        "nobs": int(len(residual)),
        "adf_stat": float(adf_stat),
        "adf_p": float(adf_p),
        "adf_crit_5pct": float(adf_crit["5%"]),
        "adf_rejects_unit_root_5pct": bool(adf_p < 0.05),
        "engle_granger_stat": float(eg_stat),
        "engle_granger_p": float(eg_p),
        "engle_granger_crit_5pct": float(eg_crit[1]),
        "cointegrated_5pct": bool(eg_p < 0.05),
        "ar1_phi": phi,
        "half_life_days": hl,
        "acf_1": float(autocorrelation[1]),
        "acf_5": float(autocorrelation[5]),
        "residual_sd": float(residual.std()),
    }


def run(panel: pd.DataFrame, curve_frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Estimate every pair and return the description, statistics and residual paths."""
    relationship_rows, statistic_rows = [], []
    residuals, zscores, betas, hedged = {}, {}, {}, {}

    for pair in PAIRS:
        y, x = pair_series(panel, curve_frame, pair)
        static, fit = static_residual(y, x)
        changes = change_relationship(y, x)
        relationship_rows.append(
            {
                "pair": pair.name,
                "y": pair.y,
                "x": pair.x,
                **fit,
                **changes,
                "sign_agrees_with_changes": bool(np.sign(fit["beta"]) == np.sign(changes["beta_changes"])),
                "primary": pair.name == PRIMARY_PAIR,
                "in_backtest": pair.name in BACKTEST_PAIRS,
                "rationale": pair.rationale,
                "tradable_leg": pair.tradable_leg,
                "tradable_note": pair.tradable_note,
            }
        )

        # Decision sample: the training split, which is what was available ex ante.
        train_y, train_x = config.slice_split(y, "train"), config.slice_split(x, "train")
        train_residual, _ = static_residual(train_y, train_x)
        statistic_rows.append(stationarity_report(train_residual, train_y, train_x, pair.name, "train"))
        statistic_rows.append(stationarity_report(static, y, x, pair.name, "full"))

        rolling = rolling_residual(y, x)
        statistic_rows.append(
            stationarity_report(rolling["residual"], y, x, pair.name, "full_rolling_beta")
        )
        residuals[pair.name] = rolling["residual"]
        zscores[pair.name] = rolling["zscore"]
        betas[pair.name] = rolling["beta"]
        if pair.return_scale is not None:
            hedged[pair.name] = spread_returns(y, x, rolling["beta"]) * pair.return_scale

    return {
        "relationships": pd.DataFrame(relationship_rows),
        "statistics": pd.DataFrame(statistic_rows),
        "residuals": pd.DataFrame(residuals),
        "zscores": pd.DataFrame(zscores),
        "betas": pd.DataFrame(betas),
        "hedged_returns": pd.DataFrame(hedged),
    }
