# Macro Markets Quantitative Research Platform

**Cross-asset macro research and relative value**
Python · pandas · NumPy · statsmodels · Excel · time series analysis

Can rates, FX, volatility, commodities, macroeconomic data and futures positioning explain
cross-asset behaviour, identify relative value dislocations, and improve short-term market
signals?

Three hypotheses, specified in advance, tested on one frozen dataset, reported whether or not
they worked.

```
FRED · ALFRED · CFTC · Cboe
        |
        v
   raw data  ──> validation ──> cleaned data ──> feature engine
                                                      |
                                    ┌─────────────────┼─────────────────┐
                                    v                 v                 v
                            H1 regressions   H2 relative value   H3 positioning
                                    └─────────────────┼─────────────────┘
                                                      v
                                                 backtester
                                                      v
                              Excel monitor · charts · research report
```

---

## Read this first: data provenance

> **The committed results were produced from a frozen synthetic dataset, not from market
> data.** The environment this project was built in had no network egress to FRED, ALFRED,
> CFTC or Cboe — all four reset at the network layer. Rather than ship a pipeline nobody had
> ever run, `src/synthetic.py` generates a fixed, seeded dataset with the same schema,
> calendar and units, so every model, backtest, chart, Excel sheet and test executes end to
> end.
>
> Every number in `outputs/` and `docs/` is a genuine output of this pipeline. **None of them
> is a finding about markets.** They describe the simulator's data-generating process. Charts
> carry a `SYNTHETIC DATA` watermark, the Excel monitor says so on its README sheet, and the
> research report opens with the same warning.
>
> The only observed market values anywhere in the project are the three anchors in
> `config.ANCHORS`, listed below. Running `python run_pipeline.py --source fred` on a machine
> with egress reproduces the identical analysis from live data.

### The anchors

The research window is frozen at **2016-08-15 to 2026-08-13**, 2,512 NYSE sessions. Three real
observations on the final date are checked on every single run, and the pipeline aborts if the
loaded data does not reproduce them:

| Anchor | Value |
|---|---|
| S&P 500 close | 7,798.99 |
| 2y Treasury yield | 4.15% |
| 10y Treasury yield | 4.63% |
| **2s10s slope** | **4.63% − 4.15% = +48 bp** |

That check is the cheapest possible guard against loading the wrong series or the wrong window,
and it is why `validation.check_anchor_values` runs before anything else.

---

## What the research found

| | Hypothesis | Result |
|---|---|---|
| **H1** | Rates, the curve and volatility explain equity returns | **Holds contemporaneously, no predictive content.** Adjusted R² **0.44** on 2,511 daily observations, all signs matching theory. Lag the same regressors one day and out-of-sample R² is **−0.009**. |
| **H2** | Cross-asset pairs produce tradable mean reversion | **Dislocations are real, monetising them is not.** **0 of 5** pairs cointegrated on the pre-registered training sample (Engle-Granger p from 0.42 to 0.98). **2 of 5** had a levels beta with the *opposite sign* to the differenced beta. |
| **H3** | Crowded futures positioning predicts returns | **Not supported.** **0 of 15** top-decile-versus-bottom-decile forward return tests reached significance on HAC standard errors. |

And the number that governs all three:

> **0 of 13 backtested strategies had a positive Sharpe in all three splits.**
>
> After Holm-Bonferroni across the 22-test family, **3 tests survive at 5% — and all three are
> contemporaneous H1 coefficients. No strategy Sharpe survives.** The best out-of-sample Sharpe
> (1.80) carries a raw p-value of 0.023 that becomes 0.41 adjusted, and the same strategy
> returned −0.76 in training.

Three findings worth the space they take up:

**The textbook equity-rates regression cannot be estimated.** Writing
`r_SPX ~ dy2 + dy10 + dCurve + dVIX` is rank deficient, because `dCurve ≡ dy10 − dy2`. Its
condition number here is **1.9 × 10¹⁵**. The platform reports it anyway, then estimates two
identified alternatives; the Fisher decomposition `y10 = real + breakeven` lifts adjusted R²
from 0.396 to 0.440, because a real-rate shock and an inflation-expectations shock have
opposite signs for equities and merging them destroys information.

**A levels regression on two trending series gives the wrong sign.** Log gold on the 10y real
yield produces beta **+0.046 (t = 7.6)** — positive, contradicting the economic prior. In first
differences the same pair gives **−0.085 (t = −22.5)**, which matches theory. Anyone reporting
only the levels beta would have published a result with the wrong sign and a t-statistic large
enough to look convincing.

**A relative-value residual is only tradable if you can hold both legs.** The primary pair's
residual does converge after an extreme reading, but most of the convergence arrives through
the real-yield leg, which is not investable in a cash universe. Trading gold alone leaves the
position wearing gold's drift, which over this window is far larger than the convergence being
harvested. Both the hedged spread and the single-leg version are reported, because the
difference between them is the finding.

Full write-up: **[`docs/RESEARCH_REPORT.md`](docs/RESEARCH_REPORT.md)**.

---

## Two leakage problems that actually mattered

**CFTC publication timing.** Commitments of Traders positions are measured at Tuesday's close
and published the following Friday at 15:30 US/Eastern. Handing Tuesday's reading to a
Wednesday model gives it three days of information that did not exist. Because 15:30 ET falls
inside the cash session, this pipeline does not assume a fill in the closing half hour either:
a report becomes usable on the **next trading session** — a six calendar day lag, seven when
that Monday is a holiday. See `features.cot_available_from`.

**Macroeconomic revisions.** CPI, payrolls, unemployment, industrial production and GDP are all
revised. A backtest reading today's 2019 CPI is using a number 2019 did not have. The loader
stores the full vintage table and reconstructs each series *as it was known* at each vintage
date before computing any transform.

**And the check that proves it.** Rather than asserting that a transform is causal,
`validation.check_causality` multiplies the last 40 rows of the input by 1.25 and requires
every earlier feature and signal value to be bit-identical. A full-sample z-score, a centred
window and a negative shift all fail it; `tests/test_leakage.py` feeds it exactly those three
to confirm the check has teeth. Current status: **55 checks, all passing.**

---

## Running it

```bash
pip install -r requirements.txt

python run_pipeline.py                 # frozen synthetic dataset, no network needed (~18s)
python run_pipeline.py --source fred   # live FRED / ALFRED / CFTC / Cboe
python -m pytest tests/ -q             # 84 tests (~7s)
python tools/build_notebooks.py        # regenerate and execute the notebooks
```

Set `FRED_API_KEY` for true ALFRED vintages. Without it a live run falls back to a release-lag
approximation and records the downgrade in the provenance file.

The pipeline exits non-zero if any critical validation check fails.

---

## Repository

```
src/
  config.py            frozen window, universe, splits, anchors — every constant, once
  synthetic.py         frozen seeded dataset (the offline fallback)
  trading_calendar.py  NYSE sessions, computed not downloaded
  data_loader.py       FRED / ALFRED / CFTC / Cboe ingestion, raw layer persistence
  validation.py        55 integrity and leakage checks, incl. the perturbation test
  features.py          returns, bp changes, z-scores, COT publication alignment, macro vintages
  regressions.py       H1: OLS with HAC errors, collinearity diagnostics, rolling betas
  relative_value.py    H2: pairs, residuals, ADF / Engle-Granger, half-lives, hedged spreads
  signals.py           signal construction and the candidate registry
  backtest.py          timing rule, costs, vol-targeted sizing, split consistency
  metrics.py           performance statistics
  regimes.py           volatility / curve / real-yield / dollar classification
  charts.py            figures, each stamped with its date range and provenance
  reporting.py         Excel monitor, CSV tables, interview sheet
  report_text.py       research report, assembled from the computed tables
  pipeline.py          orchestration: load → validate → research → backtest → report
notebooks/             01 overview · 02 relationships · 03 relative value · 04 positioning · 05 backtests
tests/                 84 tests, including leakage checks that must fail on leaky input
outputs/
  charts/              17 figures
  tables/              18 CSV tables
  macro_market_monitor.xlsx   24-sheet cross-asset monitor
docs/
  RESEARCH_REPORT.md   the full write-up
  INTERVIEW_SHEET.md   the twenty numbers, generated not typed
  DATA_SOURCES.md      every endpoint, identifier and known coverage problem
```

### The Excel monitor

`outputs/macro_market_monitor.xlsx`, 24 sheets: market snapshot, cross-asset changes over
1d/1w/1m/3m/12m, rolling relationships, positioning, current signal states, H1 regressions and
diagnostics, relative-value relationships and stationarity, positioning buckets, the signal
registry, backtest summary, split consistency, cost sensitivity, regime performance, the
multiple-testing table, the validation log, the interview numbers, the data dictionary and the
chart manifest.

Every instrument shows **its own last observation date**, so a stale series cannot be compared
against a fresh one without it being visible.

---

## Scope and methodology decisions

Choices a reviewer will want to see stated rather than inferred.

**There are no futures prices.** No Bloomberg, Refinitiv or CME settlement data, so no
continuous contracts are constructed and none are claimed. The accurate description is cash and
spot series for the cross-asset research layer, and CFTC futures-and-options positioning for
the derivatives layer. Where a futures return is unavoidable a documented proxy is used — the
10y note as `−7.8 × dy`, the euro as the negated broad-dollar return — labelled everywhere it
appears, roll yield excluded. There is no roll convention because there is nothing to roll.

**Pairs were chosen from economics, not from Sharpe ratios.** Each of the five carries the prior
that motivated it in `relative_value.PAIRS`, and the primary pair was fixed on the strength of
its theory before any backtest ran.

**Stationarity was assessed on the training split**, because that is the sample that existed
when the strategy was specified. The full-sample result is reported next to it for completeness,
not as the decision rule. The residual from a *rolling* hedge ratio does reject the unit root,
but that is substantially mechanical — a trailing regression re-centres its own residual every
day — so this project does not claim cointegration.

**Returns are excess returns.** Outright positions in a cash asset are charged financing at the
effective fed funds rate, applied to the asset return before positions so nothing is charged on
flat days. Hedged spreads and futures-proxy returns are already excess returns and are not
charged twice. Over this window the cash rate ranged from near zero to above 5%, so treating it
as zero would have flattered every result from the hiking cycle.

**Positions are volatility-targeted at 10%**, sized from trailing 63-day realised volatility,
lagged one day, capped at 3x, and **fixed at entry** for the life of the trade. Without this,
reported volatility measures which leg happens to be volatile rather than signal quality.
Rescaling daily would change the position every session and make turnover and trade counts
meaningless.

**Costs are a sensitivity, not a model.** 0, 1, 2 and 5 bp per unit traded. No spread, market
impact or borrow cost is sourced per instrument, and the report does not pretend otherwise.

**Regime labels use full-sample breakpoints** and are therefore ex-post attribution, not
tradable filters. A regime-conditioned strategy built on full-sample breakpoints is a lookahead
bug with a respectable name.

**Hit rate is measured over days holding risk**, not all days. Measured over all days, a
strategy that is flat half the time cannot exceed a 50% hit rate however good it is.

Known limitations are collected in section 12 of the research report.

---

## The same project, three ways

**To an engineer.** A Python research pipeline ingesting daily, weekly and monthly data from
several sources. Calendars and units are normalised, the raw layer is written once and never
mutated, feature and backtesting modules are reusable, and 55 automated checks cover missing
data, duplicate and future timestamps, non-positive prices, the COT open-interest identity,
vintage ordering and timestamp leakage. One issue worth the walkthrough: CFTC positions refer
to Tuesday but are not published until Friday afternoon, so they are aligned to when they
actually became available.

**To a quant.** The goal was to test whether cross-asset macro variables carry explanatory or
predictive information, not to assume a strategy exists. Prices become log returns and yields
basis point changes; the equity regression is estimated with Newey-West errors after
Breusch-Pagan and Ljung-Box reject the classical assumptions, and in a level/slope
parameterisation because the textbook specification is rank deficient. Relative-value residuals
come from economically motivated pairs with trailing hedge ratios, tested with ADF and
Engle-Granger and characterised by AR(1) half-life before any holding horizon was chosen. The
train/validation/test split was fixed in advance, positioning is lagged to its actual release,
macro data is vintage-selected, and the full candidate-signal family is declared so the
Holm-Bonferroni adjustment covers it rather than the survivors.

**To a recruiter.** A quantitative research platform covering interest rates, currencies,
equities, commodities, volatility and derivatives positioning. It uses ten years of data to
study how these markets interact, builds models to identify unusual relative value
relationships, tests those ideas historically, and produces an Excel monitor summarising
current market conditions and results.

---

## Resume lines, with the numbers filled in

Every figure below is regenerated by `python run_pipeline.py` and traceable to a file in
`outputs/tables/`. On live data they will differ; the pipeline prints its own.

> **Macro Markets Quantitative Research** | Python, pandas, NumPy, statsmodels, Excel, Time Series Analysis
>
> - Analysed **23 market and macroeconomic series** (13 daily cross-asset, 5 vintage macro, 5
>   CFTC positioning markets) across **2,512 trading days** of U.S. rates, FX, equity indices,
>   commodities, volatility and derivatives positioning, engineering **164 features** and
>   evaluating cross-asset relationships through rolling statistics and time series regression
>   with Newey-West standard errors.
> - Developed and backtested **13 relative value and positioning signals** from a declared
>   **14-signal** candidate family, applying rolling OLS, z-score normalisation, ADF and
>   Engle-Granger testing, AR(1) half-life estimation and walk-forward evaluation over a
>   pre-registered train/validation/test split; found **no strategy positive across all three
>   splits** and **no strategy Sharpe surviving** Holm-Bonferroni correction over the 22-test
>   family, while the contemporaneous rates-equity relationship held at **adjusted R² 0.44**
>   with **predictive R² of −0.009**.
> - Eliminated lookahead bias by aligning CFTC positioning to its actual Friday 15:30 ET
>   release, using ALFRED point-in-time macro vintages, and building **55 automated validation
>   checks** including a perturbation test that proves no feature or signal uses forward
>   information.
> - Built an automated **24-sheet Excel and Python market monitor** tracking yield curves,
>   rolling correlations, futures positioning percentiles and z-scores, regime classifications,
>   signal levels and strategy performance across volatility, curve, real-yield and dollar
>   regimes.

The most useful thing on this list is the third and fourth bullet, and the second bullet's
negative result. A project that reports what did not work is a project whose positive claims can
be believed.
