#!/usr/bin/env python3
"""
build_panel.py  --  fetch REAL public data for the gas-prices / driving analysis.

Run this on a machine with normal internet access (the analysis container used to
prepare the report cannot reach FRED/EIA/NHTSA, which is why the shipped
gas_driving_panel.dta is synthetic). This script pulls the national monthly
series that need no API key and writes a real, ready-to-run dataset:

    gas_national_real.csv   and   gas_national_real.dta

That real national file supports the time-series / cointegration / ECM portion
of gas_driving_analysis.do (the cleanest "do people drive less when gas is
expensive" relationship). Assembling the full STATE x MONTH panel for the
fixed-effects, count, and threshold models additionally requires FARS and UCR,
which are large file downloads rather than one-line API calls; the steps are
documented at the bottom of this file.

Sources (all free, public):
  GASREGM              EIA U.S. regular gasoline price, monthly $/gal  (via FRED)
  TRFVOLUSM227NFWA     FHWA vehicle miles traveled, monthly, millions   (via FRED)
  CPIAUCSL             BLS CPI-U, monthly index                         (via FRED)

No API key is required: we use FRED's public CSV endpoint.

Stata usage of the real national file (national time-series ECM):
    import delimited "gas_national_real.csv", clear varnames(1)
    gen mdate = mofd(date(date,"YMD")) // if 'date' is a string yyyy-mm-dd
    format mdate %tm
    tsset mdate
    gen lprice = ln(price)
    gen lvmt   = ln(vmt)
    * unit roots, then a simple ECM:
    dfuller lprice, lags(12)
    dfuller lvmt, trend lags(12)
    reg lvmt c.trend lprice
    predict ect, resid
    reg D.lvmt D.lprice L.ect, vce(robust)   // L.ect coefficient < 0 => mean reversion
"""

import io
import sys
import pandas as pd

try:
    import requests
    def _get(url):
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        return r.text
except Exception:
    from urllib.request import urlopen
    def _get(url):
        with urlopen(url, timeout=60) as resp:
            return resp.read().decode("utf-8")

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"

def fred_series(sid, colname):
    """Download one FRED series as a monthly pandas Series."""
    text = _get(FRED_CSV.format(sid=sid))
    df = pd.read_csv(io.StringIO(text))
    # FRED CSVs have columns: observation_date (or DATE), <SID>
    date_col = df.columns[0]
    val_col = df.columns[1]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df = df.dropna()
    s = df.set_index(date_col)[val_col].rename(colname)
    # collapse to month-end period mean (GASREGM/CPI are monthly already; harmless)
    s = s.groupby(s.index.to_period("M")).mean()
    return s

def main(start="2005-01", end="2024-12"):
    print("Downloading national series from FRED (no key needed)...")
    price = fred_series("GASREGM", "price")             # $/gal, monthly
    vmt   = fred_series("TRFVOLUSM227NFWA", "vmt")       # millions of miles, monthly
    cpi   = fred_series("CPIAUCSL", "cpi")               # index, monthly

    df = pd.concat([price, vmt, cpi], axis=1)
    df = df.loc[start:end].dropna()
    df = df.reset_index().rename(columns={"index": "period"})
    df["date"] = df["period"].dt.to_timestamp().dt.strftime("%Y-%m-%d")
    df["ym"] = df["period"].apply(lambda p: (p.year - 1960) * 12 + (p.month - 1)).astype(int)
    df["rprice"] = df["price"] / (df["cpi"] / df["cpi"].iloc[0] * 100) * 100  # real, base=first obs
    out = df[["date", "ym", "price", "rprice", "vmt", "cpi"]].copy()

    out.to_csv("gas_national_real.csv", index=False)
    try:
        out.to_stata("gas_national_real.dta", write_index=False, version=118,
                     data_label="Real national gas price / VMT / CPI (FRED)")
        wrote = "gas_national_real.csv and gas_national_real.dta"
    except Exception as e:
        wrote = "gas_national_real.csv (Stata export skipped: %s)" % e

    print("Wrote %s : %d months, %s to %s" %
          (wrote, len(out), out["date"].iloc[0], out["date"].iloc[-1]))
    print("Latest price in file: $%.2f/gal" % out["price"].iloc[-1])
    return out

if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) == 2:
        main(a[0], a[1])
    else:
        main()

# ============================================================================
# Building the full STATE x MONTH panel (for the FE / count / threshold models)
# ----------------------------------------------------------------------------
# These require file downloads, not one-line API calls, so they are documented
# rather than executed here. Target the same variable names as the shipped file:
#   state ym price cpi vmt fatal spdfatal duifatal urate pcinc etohpc beertax gdlyear
#
# 1. PRICES by state-month:
#    EIA state retail gasoline series (weekly) -> collapse to monthly mean.
#    https://www.eia.gov/petroleum/gasdiesel/  (or PADD-level if state is missing)
#
# 2. VMT by state-month:
#    FHWA Traffic Volume Trends monthly state tables.
#    https://www.fhwa.dot.gov/policyinformation/travel_monitoring/tvt.cfm
#
# 3. FATALITIES (total, speeding, alcohol) by state-month:
#    NHTSA FARS annual files (one zip per year). For each year, read ACCIDENT.csv
#    (or accident.sas7bdat); keep STATE, MONTH; build:
#       fatal    = count of fatal crashes (or sum FATALS)
#       spdfatal = subset where SP (speeding-related) / SPEEDREL flag is set
#       duifatal = subset where DRUNK_DR > 0 (or person-level BAC >= 0.08)
#    https://www.nhtsa.gov/file-downloads?p=nhtsa/downloads/FARS/
#    Aggregate to state x month and stack years 2005-2024.
#
# 4. DUI ARRESTS (alternative DUI measure), state x year:
#    FBI Crime Data Explorer API (free key) or UCR arrest tables.
#    https://cde.ucr.cjis.gov/   (interpolate to monthly if needed)
#
# 5. CONTROLS:
#    urate   : BLS LAUS state unemployment, on FRED as <ST>UR (e.g. CAUR, TXUR).
#    pcinc   : BEA SAINC per-capita personal income (quarterly -> monthly).
#    etohpc  : NIAAA apparent per-capita alcohol consumption (annual, state).
#    beertax : Tax Foundation beer excise ($/gal), time-invariant per state.
#    gdlyear : IIHS graduated-licensing phase-in year.
#
# Merge all on (state, ym), then run gas_driving_analysis.do unchanged.
# ============================================================================
