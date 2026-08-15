"""Hypothesis 1: do rates, the curve and volatility explain short-term equity returns?

The regression is estimated with Newey-West (HAC) standard errors because daily equity
returns are heteroskedastic and mildly autocorrelated; OLS standard errors would be
too small and the t-statistics correspondingly too flattering.

Three specifications are reported:

``naive``   dy2 + dy10 + dcurve + dvix. This is the specification as usually written
            down, and it is exactly rank deficient: dcurve = dy10 - dy2 by construction.
            It is estimated anyway, and its condition number reported, because the
            failure is the point.
``slope``   d(level) + d(slope) + dvix, where level = (y2 + y10)/2 and slope = y10 - y2.
            This spans the same information as (dy2, dy10) with no exact collinearity.
``real``    d(real 10y) + d(breakeven) + d(slope) + dvix. Uses the Fisher decomposition
            y10 = real10 + breakeven to separate a real-rate shock from an inflation
            expectations shock, which have opposite implications for equities.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

from . import config

TARGET = "ret_1d_spx"

SPECIFICATIONS: dict[str, list[str]] = {
    "naive": ["dy_1d_dgs2", "dy_1d_dgs10", "d_1d_curve_2s10s", "ret_1d_vix"],
    "slope": ["dy_1d_level", "dy_1d_slope", "ret_1d_vix"],
    "real": ["dy_1d_real10", "dy_1d_breakeven10", "dy_1d_slope", "ret_1d_vix"],
}
PRIMARY_SPEC = "real"


def hac_lags(nobs: int) -> int:
    """Newey-West bandwidth, floor(4 (n/100)^(2/9))."""
    return int(np.floor(4.0 * (nobs / 100.0) ** (2.0 / 9.0)))


def _design(features: pd.DataFrame, regressors: list[str], target: str = TARGET) -> tuple[pd.Series, pd.DataFrame]:
    data = features[[target, *regressors]].dropna()
    return data[target], sm.add_constant(data[regressors])


def fit_ols(
    features: pd.DataFrame, regressors: list[str], target: str = TARGET, label: str = ""
) -> tuple[pd.DataFrame, dict]:
    """Fit one specification and return (coefficient table, diagnostics)."""
    y, X = _design(features, regressors, target)
    lags = hac_lags(len(y))

    # The rank-deficient specification is estimated deliberately, so statsmodels' warning
    # about it is captured and reported as a diagnostic rather than printed as noise.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})

        coefficients = pd.DataFrame(
            {
                "spec": label,
                "term": model.params.index,
                "coefficient": model.params.to_numpy(),
                "std_error_hac": model.bse.to_numpy(),
                "t_stat": model.tvalues.to_numpy(),
                "p_value": model.pvalues.to_numpy(),
                "nobs": int(model.nobs),
            }
        ).reset_index(drop=True)

        exog = X.drop(columns="const")
        # Variance inflation factors from the inverse correlation matrix; a singular
        # design matrix is reported as infinite rather than raising.
        try:
            vif = pd.Series(np.diag(np.linalg.inv(exog.corr().to_numpy())), index=exog.columns)
        except np.linalg.LinAlgError:
            vif = pd.Series(np.inf, index=exog.columns)

        residuals = model.resid
        diagnostics = {
            "spec": label,
            "nobs": int(model.nobs),
            "r_squared": float(model.rsquared),
            "adj_r_squared": float(model.rsquared_adj),
            "f_pvalue": float(model.f_pvalue),
            "hac_lags": lags,
            "condition_number": float(np.linalg.cond(X.to_numpy())),
            "max_vif": float(vif.max()),
            "durbin_watson": float(durbin_watson(residuals)),
            "breusch_pagan_p": float(het_breuschpagan(residuals, X)[1]),
            "ljung_box_p_10": float(acorr_ljungbox(residuals, lags=[10], return_df=True)["lb_pvalue"].iloc[0]),
            "residual_skew": float(pd.Series(residuals).skew()),
            "residual_kurtosis": float(pd.Series(residuals).kurtosis()),
            "rank_deficient": bool(np.linalg.matrix_rank(X.to_numpy()) < X.shape[1]),
            "fit_warnings": "; ".join(sorted({str(w.message).split(".")[0] for w in caught})),
        }
    return coefficients, diagnostics


def run_specifications(features: pd.DataFrame, sample: str = "full") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate every specification on one sample."""
    frame = features if sample == "full" else config.slice_split(features, sample)
    coefficient_tables, diagnostic_rows = [], []
    for label, regressors in SPECIFICATIONS.items():
        coefficients, diagnostics = fit_ols(frame, regressors, label=label)
        coefficients["sample"] = sample
        diagnostics["sample"] = sample
        coefficient_tables.append(coefficients)
        diagnostic_rows.append(diagnostics)
    return pd.concat(coefficient_tables, ignore_index=True), pd.DataFrame(diagnostic_rows)


def rolling_betas(features: pd.DataFrame, spec: str = PRIMARY_SPEC, window: int = 252) -> pd.DataFrame:
    """Rolling window coefficients, to show whether the relationship is stable."""
    y, X = _design(features, SPECIFICATIONS[spec])
    fitted = RollingOLS(y, X, window=window).fit()
    params = fitted.params.dropna()
    tvalues = (fitted.params / fitted.bse).loc[params.index]
    params.columns = [f"beta_{c}" for c in params.columns]
    tvalues.columns = [f"t_{c}" for c in tvalues.columns]
    return pd.concat([params, tvalues], axis=1)


def coefficient_stability(rolling: pd.DataFrame, spec: str = PRIMARY_SPEC) -> pd.DataFrame:
    """Summarise each rolling coefficient path: dispersion and sign consistency."""
    rows = []
    for column in [c for c in rolling.columns if c.startswith("beta_") and c != "beta_const"]:
        path = rolling[column].dropna()
        share_positive = float((path > 0).mean())
        rows.append(
            {
                "spec": spec,
                "term": column.removeprefix("beta_"),
                "mean": float(path.mean()),
                "std": float(path.std()),
                "min": float(path.min()),
                "max": float(path.max()),
                "share_positive": share_positive,
                "sign_stable": bool(max(share_positive, 1 - share_positive) >= 0.95),
                "windows": int(len(path)),
            }
        )
    return pd.DataFrame(rows)


def regime_regressions(features: pd.DataFrame, regimes: pd.DataFrame, spec: str = PRIMARY_SPEC) -> pd.DataFrame:
    """Re-estimate the primary specification inside each regime bucket."""
    regressors = SPECIFICATIONS[spec]
    rows = []
    for dimension in regimes.columns:
        for bucket in regimes[dimension].dropna().unique():
            mask = regimes[dimension] == bucket
            subset = features.loc[mask[mask].index]
            if len(subset.dropna(subset=[TARGET, *regressors])) < 100:
                continue
            coefficients, diagnostics = fit_ols(subset, regressors, label=spec)
            for _, row in coefficients.iterrows():
                if row["term"] == "const":
                    continue
                rows.append(
                    {
                        "dimension": dimension,
                        "regime": bucket,
                        "term": row["term"],
                        "coefficient": row["coefficient"],
                        "t_stat": row["t_stat"],
                        "p_value": row["p_value"],
                        "adj_r_squared": diagnostics["adj_r_squared"],
                        "nobs": diagnostics["nobs"],
                    }
                )
    return pd.DataFrame(rows)


def out_of_sample(features: pd.DataFrame, spec: str = PRIMARY_SPEC) -> pd.DataFrame:
    """Two out-of-sample exercises, which answer different questions.

    ``contemporaneous`` fits on the training split and measures explanatory R^2 on the
    later splits using same-day regressors. It tests whether the *relationship* is
    stable out of sample. It is not tradable: same-day yield and VIX changes are not
    known when the equity return is realised.

    ``predictive`` lags every regressor by one day, so the model forecasts tomorrow's
    return from information available today. This is the tradable version, and a
    negative R^2 means the model is worse than the training-sample mean.
    """
    regressors = SPECIFICATIONS[spec]
    rows = []
    for mode in ("contemporaneous", "predictive"):
        frame = features[[TARGET, *regressors]].copy()
        if mode == "predictive":
            frame[regressors] = frame[regressors].shift(1)
        frame = frame.dropna()

        train = config.slice_split(frame, "train")
        model = sm.OLS(train[TARGET], sm.add_constant(train[regressors])).fit()
        for split in ("train", "validation", "test"):
            subset = config.slice_split(frame, split)
            if subset.empty:
                continue
            predicted = model.predict(sm.add_constant(subset[regressors]))
            actual = subset[TARGET]
            residual_ss = float(((actual - predicted) ** 2).sum())
            total_ss = float(((actual - train[TARGET].mean()) ** 2).sum())
            rows.append(
                {
                    "mode": mode,
                    "spec": spec,
                    "split": split,
                    "nobs": int(len(subset)),
                    "r_squared": 1.0 - residual_ss / total_ss,
                    "sign_accuracy": float((np.sign(predicted) == np.sign(actual)).mean()),
                    "correlation": float(np.corrcoef(predicted, actual)[0, 1]),
                }
            )
    return pd.DataFrame(rows)


def holm_bonferroni(p_values: pd.Series) -> pd.DataFrame:
    """Holm-Bonferroni step-down adjustment over the project's headline tests.

    Three hypotheses tested with a handful of specifications each is a small family, but
    it is not one test, and the adjustment belongs in the write-up rather than in a
    reviewer's question.
    """
    ordered = p_values.dropna().sort_values()
    n = len(ordered)
    adjusted, running = [], 0.0
    for rank, value in enumerate(ordered.to_numpy()):
        running = max(running, min(1.0, (n - rank) * value))
        adjusted.append(running)
    return pd.DataFrame(
        {
            "test": ordered.index,
            "p_value": ordered.to_numpy(),
            "p_holm": adjusted,
            "significant_5pct": [a < 0.05 for a in adjusted],
            "family_size": n,
        }
    )
