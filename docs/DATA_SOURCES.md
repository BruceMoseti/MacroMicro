# Data sources

Five datasets, and the exact request each one makes. Every identifier used anywhere in the
project is declared in `src/config.py`, not scattered through the code.

> **The live endpoints below were never exercised.** The environment this project was built
> in had no network egress to `fred.stlouisfed.org`, `alfred.stlouisfed.org`,
> `publicreporting.cftc.gov` or `cdn.cboe.com`; all four reset at the network layer. The
> request shapes follow each provider's documented API and should be confirmed on a first
> live run. The synthetic path in `src/synthetic.py` is fully exercised by the test suite and
> by every artifact in `outputs/`.

## 1. FRED daily market series

Endpoint: `https://fred.stlouisfed.org/graph/fredgraph.csv?id=<ID>&cosd=<start>&coed=<end>`

No API key required. Missing observations are encoded as `.` and are coerced to NaN rather
than forward-filled at load time, so `validation.check_daily_panel` can count and document
them.

| Column | FRED ID | Description | Unit | Transform |
|---|---|---|---|---|
| `dgs2` | `DGS2` | 2y Treasury constant maturity | percent | bp change |
| `dgs10` | `DGS10` | 10y Treasury constant maturity | percent | bp change |
| `dgs30` | `DGS30` | 30y Treasury constant maturity | percent | bp change |
| `real10` | `DFII10` | 10y TIPS real yield | percent | bp change |
| `breakeven10` | `T10YIE` | 10y breakeven inflation | percent | bp change |
| `fedfunds` | `DFF` | Effective federal funds rate | percent | level (financing) |
| `usd` | `DTWEXBGS` | Nominal broad USD index, Jan-2006 = 100 | index | log return |
| `spx` | `SP500` | S&P 500 close | index | log return |
| `ndx` | `NASDAQ100` | Nasdaq 100 close | index | log return |
| `vix` | `VIXCLS` | Cboe VIX close | index | log return |
| `wti` | `DCOILWTICO` | WTI crude, Cushing OK | USD/bbl | log return |
| `gold` | `GOLDPMGBD228NLBM` | LBMA gold price, PM fix | USD/oz | log return |
| `hy_oas` | `BAMLH0A0HYM2` | ICE BofA US high yield OAS | percent | bp change |

### Known coverage problems

Both of these are handled explicitly rather than discovered as silent NaNs.

- **Negative WTI.** `DCOILWTICO` printed **-$37.63 on 2020-04-20**. A log return is undefined
  on a non-positive price, so `validation.check_daily_panel` fails the run on any non-positive
  price rather than propagating `NaN` or `-inf` through the feature engine. A live run has to
  make an explicit decision here (floor the print, treat the day as missing, or switch to a
  front-month futures series); the frozen synthetic scenario floors the trough at the lowest
  positive print and says so in `src/synthetic.py`.
- **Discontinued gold series.** FRED's LBMA gold price was discontinued, so the tail of the
  window needs an alternate source. The gap is recorded in
  `config.Series.coverage_note`, surfaced in the dataset provenance record, and printed in the
  Excel monitor's README sheet.

## 2. ALFRED vintage macroeconomic series

Endpoint: `https://api.stlouisfed.org/fred/series/observations` with
`realtime_start=1776-07-04`, `realtime_end=9999-12-31`, `output_type=1`.

Requires `FRED_API_KEY` in the environment. Each observation returns with a `realtime_start`,
which is the date that value first became public; that is stored as the vintage date.

| Column | FRED ID | Frequency | Release lag fallback |
|---|---|---|---|
| `cpi` | `CPIAUCSL` | monthly | 14 days after period end |
| `payrolls` | `PAYEMS` | monthly | 6 days |
| `unrate` | `UNRATE` | monthly | 6 days |
| `indpro` | `INDPRO` | monthly | 16 days |
| `gdp` | `GDPC1` | quarterly | 29 days |

**Without an API key** the loader falls back to the current (revised) series stamped with the
conservative release lag above. That removes publication-lag leakage but **not** revision
bias, and the downgrade is written into the provenance record so it appears in the research
report rather than being quietly assumed away.

`features.macro_point_in_time` reconstructs each series *as it was known* at every vintage
date before computing any transform, so a year-over-year inflation rate in a 2019 backtest is
the number that was actually printed in 2019.

## 3-4. CFTC Commitments of Traders

Endpoint: `https://publicreporting.cftc.gov/resource/<dataset>.json` (Socrata), filtered on
`report_date_as_yyyy_mm_dd` and `market_and_exchange_names`.

| Report | Dataset | Markets | Speculative category |
|---|---|---|---|
| Traders in Financial Futures | `gpe5-46if` | E-mini S&P 500, 10y Treasury note, Euro FX | leveraged funds |
| Disaggregated | `72hh-3qpy` | WTI crude, Gold | managed money |

Dataset identifiers and column names are in `config` and `data_loader.CFTC_COLUMN_MAP`; verify
them against the live catalogue at <https://publicreporting.cftc.gov> on first use. Both
reports are normalised into one tidy schema — `market, report, contract_name, report_date,
category, long, short, open_interest` — so downstream code has a single shape to handle.

### Publication timing

Positions are as of **Tuesday's close** and are published the following **Friday at 15:30
US/Eastern**. Because 15:30 ET falls inside the cash session, the pipeline does not assume a
fill in the closing half hour: a report becomes usable on the **next trading session**, a six
calendar day lag from the Tuesday as-of date and seven when that Monday is a holiday. See
`features.cot_available_from`.

## 5. Cboe VIX history

Endpoint: `https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv`

Used as an independent cross-check on FRED's `VIXCLS` rather than as the primary source, so a
silent change in either provider's series shows up as a disagreement.

## What is deliberately absent

No futures price series. There is no Bloomberg, Refinitiv or CME settlement data here, so the
project does not construct continuous contracts and does not claim to. The honest description
is **cash and spot series for the cross-asset research layer, and CFTC futures-and-options
positioning for the derivatives layer.**

Where a futures return is unavoidable, a documented proxy is used and labelled everywhere it
appears:

| Market | Proxy | Excluded |
|---|---|---|
| ES | cash S&P 500 log return | basis, dividends, roll |
| ZN | -7.8 x d(10y yield) | convexity, carry, roll |
| 6E | negated broad USD index return | rate differential, roll |
| CL | WTI spot log return | roll yield, storage |
| GC | gold spot log return | roll yield, lease rate |

These are in `config.COT_UNIVERSE[*].proxy_note` and are reproduced in the Excel monitor's
data dictionary.
