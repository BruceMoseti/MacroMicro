# Numbers I must know

Generated from `outputs/tables/interview_numbers.csv`. Data source: **SYNTHETIC**  
**These figures come from the synthetic dataset and are not market history.**

**1. Datasets**  
5: FRED daily market series; ALFRED vintage macroeconomic series; CFTC Traders in Financial Futures (COT); CFTC Disaggregated Futures (COT); Cboe VIX history

**2. Instruments and series**  
13 daily market series, 5 vintage macro series, 5 CFTC positioning markets = 23 total; 164 engineered features

**3. Daily observations**  
2,512 trading days; 32,656 daily series-observations; 522 weekly COT reports; 1,833 macro vintage rows

**4. Start and end date**  
2016-08-15 to 2026-08-13 (frozen). Train to 2022-12-31, validation to 2024-12-31, test from 2025-01-01

**5. Candidate signals tested**  
14 declared in signals.SIGNAL_REGISTRY, of which 13 were backtested; 3 regression specifications; 5 relative-value pairs; 5 positioning markets

**6. Actual trades**  
1,596 across all backtested signals, full sample, at 2bp

**7. Best out-of-sample Sharpe**  
1.80 (h2_rv_gold_real10_hedged, 404 days, 21 trades, 2bp costs). Say the next sentence unprompted: 0 of 13 strategies were positive in all three splits, so this number is almost certainly noise, not an edge

**8. Maximum drawdown**  
-4.4% for the primary strategy out of sample; -69.3% worst across all signals (full sample)

**9. Hit rate**  
53.4% for the primary strategy out of sample, over 118 days holding risk

**10. Average holding period**  
5.7 trading days out of sample; 12.7 days on average across signals

**11. Transaction cost assumption**  
sensitivity grid 0, 1, 2, 5bp per unit traded, headline 2bp. Not a sourced cost model. Financing at the effective fed funds rate is charged on outright cash positions only

**12. Most important regression coefficient**  
ret_1d_vix = -0.07211 (HAC t = -14.6, p = 2.70e-48, n = 2511). A 1% rise in VIX coincides with a -7.21bp move in the S&P 500.

**13. Regression R-squared**  
adjusted R2 = 0.440 contemporaneous (spec 'real'); predictive adjusted R2 is approximately zero (see h1_out_of_sample)

**14. Stationarity result**  
gold_real10 training-sample residual: ADF = -0.32 (p = 0.923), Engle-Granger p = 0.975, AR(1) phi = 0.9997, half-life = 1987 days. No pair was cointegrated at 5%

**15. Best performing regime**  
real_yield = top quartile (>1.48%), Sharpe 0.80 over 628 days

**16. Worst performing regime**  
real_yield = middle 50% (-0.03-1.48%), Sharpe -0.86 over 1130 days

**17. Biggest failure**  
H1 has no predictive content. The contemporaneous regression explains 44% of daily equity variance, but lagging the same regressors by one day gives a negative out-of-sample R2, so the relationship is an accounting identity of same-day repricing rather than a forecast.

**18. Biggest source of potential bias**  
Multiple testing and pair selection. Five relative-value pairs, three regression specifications and five positioning markets were evaluated on one fixed sample; the Holm-Bonferroni adjustment over the 22-test family is reported rather than the raw p-values.

**19. Futures roll handling**  
No futures price series is used, so there is no roll. Derivatives enter through CFTC futures-and-options positioning. Return proxies for the positioning markets are cash or duration-approximated and exclude roll yield, which is stated wherever they appear.

**20. Lookahead prevention**  
Positions at t earn the return from t to t+1; COT positions are Tuesday readings withheld until the session after Friday's 15:30 ET release (a 6 day lag); macro data is vintage-selected point-in-time; every rolling statistic and hedge ratio uses a trailing window; and a perturbation test confirms that changing the last 40 rows of input leaves all earlier features and signals bit-identical.
