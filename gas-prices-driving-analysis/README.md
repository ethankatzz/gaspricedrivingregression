# Gas Prices and Driving Behavior

An econometric study of how United States gasoline prices affect three margins of driving behavior: how much people drive (vehicle miles traveled), drunk driving (DUI), and speeding. The repository contains a complete Stata workflow, a Python replication engine, a ready-to-run panel dataset, and the regression and diagnostic output.

The work is motivated by the 2026 price episode (the national average rose from about $2.81 a gallon in January to roughly $4.55 to $4.60 at the late-May peak, around $4.29 as of early June, driven by a geopolitical supply shock), which is a textbook exogenous, supply-driven price move.

## Headline findings

These conclusions come from the peer-reviewed literature (see Sources). The dataset shipped here is synthetic and calibrated to those findings, so running the code reproduces the same signs and magnitudes.

- **Driving (VMT): less, but inelastic.** Short-run elasticity about -0.05 to -0.10, rising to -0.15 to -0.25 over a year or two. A 50 percent price increase implies very roughly a 3 to 6 percent near-term cut in miles, concentrated in discretionary trips.
- **Drunk driving (DUI): less.** Higher prices mean fewer alcohol-impaired crashes, mainly through reduced exposure (fewer miles), concentrated in less severe crashes.
- **Speeding: roughly no change.** Drivers do not slow to save fuel, but emptier roads get faster (decongestion), so the per-mile speeding-crash rate rises while exposure falls, and the two effects on total counts nearly cancel.
- **Total fatalities: fall**, on the order of 2 percent per 10 percent of sustained price increase, with young drivers most affected.
- **Thresholds:** the structural relationship is closer to a constant elasticity than to a hard kink, with a behavioral inflection near $4 a gallon.

## Repository layout

```
gas-prices-driving-analysis/
  README.md
  requirements.txt
  .gitignore
  code/
    gas_driving_analysis.do    Stata workflow (built-in commands only, no add-ons)
    replication.py             Python engine: builds the panel, runs all models, makes figures
    build_panel.py             downloads real national series from FRED (no API key)
  data/
    gas_driving_panel.dta      synthetic state x month panel, 51 states x 240 months
    gas_driving_panel.csv      same data as CSV
  output/
    results.json               every coefficient, standard error, and test statistic
    figures/                   six analysis figures (PNG)
```

## Data note (please read)

The file `data/gas_driving_panel.dta` (and the CSV) is **synthetic, calibrated data**. Its data-generating process is set so the estimators recover the elasticities documented in the literature. It exists so the code runs end to end and reproduces the expected output. It is **not** scraped from official sources and is not an independent empirical estimate.

To run on **real data**:

- `code/build_panel.py` downloads the real national gasoline price (EIA `GASREGM`), vehicle miles traveled (FHWA `TRFVOLUSM227NFWA`), and CPI (`CPIAUCSL`) from FRED with no API key, and writes `gas_national_real.csv` / `.dta`. That supports the national time-series and error-correction portion.
- The full state-by-month panel for the fixed-effects, count, and threshold models additionally requires NHTSA FARS fatality files and FBI UCR arrests, which are bulk downloads rather than one-line pulls. The exact series, URLs, and aggregation steps are documented at the bottom of `build_panel.py`. Build a file with the same variable names and the Stata and Python code run unchanged.

## How to run

### Stata (17 or later; tested on Stata 19)

The do-file uses only official built-in commands, so nothing needs to be installed.

1. Open `code/gas_driving_analysis.do`.
2. Set Stata's working directory to the repository root (File > Change Working Directory, or `cd "path/to/gas-prices-driving-analysis"`).
3. Run the file.

It loads `data/gas_driving_panel.dta` and prints, in sequence: the panel fixed-effects elasticities, a Hausman test, a 12-month distributed lag with the cumulative elasticity, negative-binomial count models with an exposure offset, a Hansen-style threshold search, quadratic and spline robustness, the national unit-root, cointegration, and error-correction analysis, and the full diagnostic battery.

### Python (3.9 or later)

```
pip install -r requirements.txt
python code/replication.py
```

This builds the synthetic panel, runs every specification, writes `output/results.json`, regenerates the six figures in `output/figures/`, and re-exports `data/gas_driving_panel.dta` and `.csv`. Paths are resolved relative to the script, so it works from any working directory.

### Real national data

```
python code/build_panel.py
```

## Methods

Panel fixed-effects elasticities (log-log, state and calendar-month effects, clustered standard errors); a finite distributed lag for the dynamics of the driving response; negative-binomial count models with log miles as an exposure offset (so the price coefficient is the per-mile rate, net of exposure); a Hansen (2000) threshold search for a "certain point"; national unit-root tests (ADF, DF-GLS, Phillips-Perron), Engle-Granger and Johansen cointegration, and a single-equation error-correction model; and a diagnostic battery (Breusch-Pagan, Breusch-Godfrey, Durbin-Watson, Ramsey RESET, variance inflation factors, Hausman, and a Poisson-versus-negative-binomial overdispersion check).

## Key results from the shipped synthetic panel

| Outcome (log) | Panel FE elasticity | Reading |
|---|---|---|
| VMT (driving) | -0.148 | inelastic, as expected |
| Total fatalities | -0.174 | fall with price |
| Alcohol-impaired (DUI) | -0.270 | fall with price |
| Speeding fatalities | -0.048 (p = 0.14) | not distinguishable from zero |

Distributed-lag cumulative VMT elasticity by twelve months: about -0.18. Negative-binomial per-mile rate: DUI -0.146, speeding +0.076 (positive, via decongestion). Threshold near $4, with the fatality-price slope steepening from about -0.17 below to about -0.47 above. National driving and price are cointegrated (Engle-Granger p = 0.005) with an error-correction adjustment speed of about -0.18 per month.

## Sources

Data: U.S. Energy Information Administration (gasoline prices); Federal Highway Administration Traffic Volume Trends (vehicle miles); NHTSA Fatality Analysis Reporting System (fatalities, speeding and alcohol subsets); FBI UCR / Crime Data Explorer (DUI arrests); BLS LAUS (unemployment); BEA (personal income); NIAAA (alcohol consumption); AAA (daily prices and consumer surveys).

Key literature: Gillingham (2014, Regional Science and Urban Economics); Hughes, Knittel and Sperling (2008, The Energy Journal); Grabowski and Morrisey (2004, Journal of Policy Analysis and Management; 2006, Economics Letters); Chi et al. (2011, Accident Analysis and Prevention); Burger and Kaffine (2009, Review of Economics and Statistics); Bento et al. (2009, American Economic Review); Hansen (2000, Econometrica).

## Disclaimer

This repository is research and analysis. It is not legal, safety, or investment advice. The shipped dataset is synthetic; substantive conclusions rest on the cited literature.
