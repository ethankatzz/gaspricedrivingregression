"""
Gas prices, driving, DUI and speeding — replication / demonstration engine.

This script (a) constructs a CALIBRATED synthetic state-month panel whose
data-generating process is set to match the elasticities reported in the
peer-reviewed literature, and (b) runs the full econometric workflow used in
the report so that real regression tables, diagnostic-test output and figures
can be produced. The synthetic data exist only to demonstrate that the Stata /
econometric code runs and to illustrate expected output; they are NOT an
independent empirical estimate. All substantive conclusions in the report come
from the cited literature, not from this file.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_breusch_godfrey, linear_reset
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller, kpss, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from linearmodels.panel import PanelOLS, RandomEffects

import os
# Resolve output locations relative to this script so the repo is portable.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE) if os.path.basename(_HERE) == "code" else _HERE
DATA_DIR = os.path.join(_ROOT, "data")
OUT_DIR  = os.path.join(_ROOT, "output")
FIG_DIR  = os.path.join(OUT_DIR, "figures")
for _d in (DATA_DIR, OUT_DIR, FIG_DIR):
    os.makedirs(_d, exist_ok=True)

RNG = np.random.default_rng(20260602)
OUT = {}

# ----------------------------------------------------------------------------
# Clean, FT-ish plotting style
# ----------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 200,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#d9d9d9",
    "grid.linewidth": 0.6,
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})
NAVY = "#13294b"
BURG = "#7a1f2b"
TEAL = "#1f6f78"
GOLD = "#b8860b"
GREY = "#8a8a8a"

# ----------------------------------------------------------------------------
# 1. National monthly real gasoline price path, 2005-01 .. 2026-05
#    Anchored to reported EIA/AAA annual averages and known episode peaks/lows.
# ----------------------------------------------------------------------------
# nominal annual averages (USD/gal regular, approx, EIA)
annual_nominal = {
    2005: 2.30, 2006: 2.59, 2007: 2.80, 2008: 3.27, 2009: 2.35, 2010: 2.79,
    2011: 3.53, 2012: 3.64, 2013: 3.53, 2014: 3.37, 2015: 2.45, 2016: 2.14,
    2017: 2.42, 2018: 2.72, 2019: 2.60, 2020: 2.17, 2021: 3.02, 2022: 3.96,
    2023: 3.52, 2024: 3.30, 2025: 3.18, 2026: 3.70,
}
months = pd.period_range("2005-01", "2026-05", freq="M")
# build a monthly nominal series by interpolating annual averages and adding
# a summer seasonal bump plus the salient episode shocks
base = pd.Series(index=months, dtype=float)
for p in months:
    base[p] = annual_nominal[p.year]
base = base.interpolate()
seasonal_bump = 0.12 * np.sin(2 * np.pi * (np.array([p.month for p in months]) - 4) / 12)
nom = base.values + seasonal_bump
# overlay known episode features
idx = {p: i for i, p in enumerate(months)}
def bump(period_str, val):
    p = pd.Period(period_str, "M")
    if p in idx:
        nom[idx[p]] = val
# 2008 spike and crash
bump("2008-07", 4.11); bump("2008-06", 4.05); bump("2008-12", 1.69); bump("2008-11", 2.15)
# 2014 late crash
bump("2014-06", 3.69); bump("2015-01", 2.12)
# COVID dip
bump("2020-04", 1.84); bump("2020-05", 1.87)
# 2022 record
bump("2022-06", 5.01); bump("2022-05", 4.55)
# 2026 Strait of Hormuz shock
bump("2026-01", 2.81); bump("2026-02", 2.96); bump("2026-03", 4.06)
bump("2026-04", 4.20); bump("2026-05", 4.56)
nom = pd.Series(nom, index=months).interpolate()

# CPI deflator (rough, index 2024=1.00) to get real prices for the context chart
cpi = pd.Series(index=months, dtype=float)
cpi_annual = {2005:0.66,2006:0.68,2007:0.70,2008:0.73,2009:0.72,2010:0.74,
              2011:0.76,2012:0.78,2013:0.79,2014:0.80,2015:0.80,2016:0.81,
              2017:0.83,2018:0.85,2019:0.86,2020:0.87,2021:0.91,2022:0.98,
              2023:1.02,2024:1.05,2025:1.08,2026:1.11}
for p in months:
    cpi[p] = cpi_annual[p.year]
cpi = cpi.interpolate()
real = nom / cpi  # real price in 2024-ish dollars

price_df = pd.DataFrame({"nominal": nom, "real": real})

# ----------------------------------------------------------------------------
# 2. Calibrated state-month panel, 2005-01 .. 2024-12 (panel period)
# ----------------------------------------------------------------------------
panel_months = pd.period_range("2005-01", "2024-12", freq="M")
T = len(panel_months)
states = [
 "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA",
 "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
 "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT",
 "VA","WA","WV","WI","WY","DC"]
S = len(states)

lnP_nat = np.log(nom.loc[panel_months].values)          # national log nominal price
month_of_year = np.array([p.month for p in panel_months])
trend = np.arange(T) / 12.0                              # years since 2005
covid = np.zeros(T)
for ystr, v in {"2020-03":-0.18,"2020-04":-0.40,"2020-05":-0.27,
                "2020-06":-0.16,"2020-07":-0.09}.items():
    pp = pd.Period(ystr,"M")
    covid[list(panel_months).index(pp)] = v
recession = ((np.array([p.year for p in panel_months])>=2008) &
             (np.array([p.year for p in panel_months])<=2010)).astype(float)

# state-level structural pieces
size_s   = RNG.normal(np.log(5.0), 0.75, S)              # log billion VMT scale
size_s[states.index("CA")] += 1.4; size_s[states.index("TX")] += 1.3
size_s[states.index("WY")] -= 1.6; size_s[states.index("DC")] -= 2.2
pdiff_s  = RNG.normal(0.0, 0.05, S)                      # log price differentials
pdiff_s[states.index("CA")] += 0.20; pdiff_s[states.index("HI")] += 0.18
beer_tax = np.abs(RNG.normal(0.30, 0.25, S))             # $/gal beer excise (time-inv)
gdl_year = RNG.integers(2004, 2012, S)                   # year graduated licensing fully phased

def seasonal(amp, phase):
    return amp * np.sin(2*np.pi*(month_of_year - phase)/12)

rows = []
for j, st in enumerate(states):
    # state price = national + differential + small AR(1) idiosyncratic
    u = np.zeros(T); e = RNG.normal(0,0.015,T)
    for t in range(1,T):
        u[t] = 0.5*u[t-1] + e[t]
    lnP_st = lnP_nat + pdiff_s[j] + u
    P_lev  = np.exp(lnP_st)

    # macro controls
    urate = (5.0 + 2.8*recession + seasonal(0.3,1) + RNG.normal(0,0.4,T)
             + np.where(np.array([p.year for p in panel_months])==2020, 4.5, 0))
    lninc = (np.log(42000) + 0.018*trend - 0.02*recession + RNG.normal(0,0.01,T)).cumsum()*0 \
            + np.log(42000) + 0.02*trend - 0.03*recession + RNG.normal(0,0.012,T)
    alcohol_pc = 2.3 + 0.15*seasonal(1,7) + 0.2*recession + RNG.normal(0,0.08,T)  # gal ethanol pc

    # ---- VMT: constant-elasticity distributed lag (short-run -0.06, cum ~ -0.20)
    dl = np.array([-0.060,-0.050,-0.040,-0.020,-0.015,-0.010,-0.005])
    price_dl = np.zeros(T)
    for k,b in enumerate(dl):
        price_dl += b*np.roll(lnP_st, k)
        price_dl[:k] = price_dl[k] if k>0 else price_dl[:1]
    lnVMT = (size_s[j] + seasonal(0.07,4) + 0.012*trend + covid
             + price_dl + 0.35*(lninc-np.log(42000)) - 0.010*urate
             + RNG.normal(0,0.020,T))
    VMT = np.exp(lnVMT)  # billion miles / month

    # ---- Total fatalities: THRESHOLD dgp (per-period elasticity -0.15 below $4,
    #      -0.40 above $4); exposure via fixed state scale, mean ~70/state-month
    thr = 4.0
    pl_low  = np.log(np.minimum(P_lev, thr))
    pl_high = np.log(np.maximum(P_lev/thr, 1.0))
    ln_mu_fat = (2.6 + size_s[j] + seasonal(0.10,7) + 0.002*trend
                 - 0.15*pl_low - 0.40*pl_high + 0.010*urate
                 + RNG.normal(0,0.04,T))
    fatalities = RNG.poisson(np.maximum(np.exp(ln_mu_fat),0.5))

    # ---- Alcohol-impaired crashes (DUI proxy): NB count, exposure via ln(VMT),
    #      per-mile rate elasticity -0.15 (exposure channel dominates); alcohol
    #      consumption raises the rate. Mean ~300/state-month (DUI-incident scale).
    ln_alc = (4.2 + np.log(VMT) + seasonal(0.16,7) + 0.001*trend
              - 0.15*lnP_st + 0.20*(alcohol_pc-2.3) + 0.010*urate)
    mu_alc = np.exp(ln_alc)
    alpha_nb = 0.35
    # NB via gamma-Poisson mixture (mean preserved, variance inflated)
    gam = RNG.gamma(1/alpha_nb, alpha_nb, T)
    alc_crashes = RNG.poisson(np.maximum(mu_alc*gam, 0.05))

    # ---- Speeding fatalities: per-mile rate RISES (+0.08) via decongestion
    #      (emptier roads, higher speeds), but exposure (fewer miles) offsets, so
    #      the NET effect on total counts is near zero. Mean ~40/state-month.
    ln_spd = (2.0 + np.log(VMT) + seasonal(0.09,7) + 0.001*trend
              + 0.08*lnP_st + 0.05*(alcohol_pc-2.3) + 0.006*urate)
    mu_spd = np.exp(ln_spd)
    gam2 = RNG.gamma(1/0.25, 0.25, T)
    speed_fat = RNG.poisson(np.maximum(mu_spd*gam2, 0.05))

    df = pd.DataFrame({
        "state": st, "month": panel_months.astype(str),
        "period": panel_months,
        "moy": month_of_year, "trend": trend,
        "lnP": lnP_st, "P": P_lev,
        "lnVMT": lnVMT, "VMT": VMT,
        "fatalities": fatalities,
        "alc_crashes": alc_crashes,
        "speed_fat": speed_fat,
        "urate": urate, "lninc": lninc, "alcohol_pc": alcohol_pc,
        "beer_tax": beer_tax[j], "gdl_year": gdl_year[j],
    })
    rows.append(df)

panel = pd.concat(rows, ignore_index=True)
panel["ln_fatal"] = np.log(panel["fatalities"].clip(lower=1))
panel["ln_alc"]   = np.log(panel["alc_crashes"].clip(lower=1))
panel["ln_speed"] = np.log(panel["speed_fat"].clip(lower=1))
panel["t_index"]  = panel.groupby("state").cumcount()
print(f"Panel built: {panel.shape[0]:,} obs, {S} states x {T} months "
      f"({panel_months[0]}..{panel_months[-1]})")
OUT["panel_obs"] = int(panel.shape[0]); OUT["panel_states"]=S; OUT["panel_months"]=T

# index for linearmodels
pidx = panel.set_index(["state","t_index"])

# month-of-year dummies + trend + controls
def add_seasonal(d):
    md = pd.get_dummies(d["moy"], prefix="m", drop_first=True).astype(float)
    return md

# ----------------------------------------------------------------------------
# 3. Panel fixed-effects (state FE + seasonal + trend + controls), clustered SE
#    NB: full year-month FE would absorb the (largely national) price, so we
#    follow Grabowski-Morrisey and use state FE + calendar-month + trend.
# ----------------------------------------------------------------------------
def panel_fe(yvar):
    md = add_seasonal(pidx)
    X = pd.concat([pidx[["lnP","trend","urate","lninc"]], md], axis=1)
    X = sm.add_constant(X)
    mod = PanelOLS(pidx[yvar], X, entity_effects=True, drop_absorbed=True, check_rank=False)
    res = mod.fit(cov_type="clustered", cluster_entity=True)
    return res

fe_results = {}
for y, lab in [("lnVMT","VMT (driving)"), ("ln_fatal","Total fatalities"),
               ("ln_alc","Alcohol-impaired crashes"), ("ln_speed","Speeding fatalities")]:
    r = panel_fe(y)
    b = r.params["lnP"]; se = r.std_errors["lnP"]; t = b/se; p = r.pvalues["lnP"]
    fe_results[lab] = dict(beta=float(b), se=float(se), t=float(t), p=float(p),
                           n=int(r.nobs), within_r2=float(r.rsquared_within))
    print(f"[FE] {lab:28s} elasticity={b:+.3f}  se={se:.3f}  t={t:+.2f}  p={p:.3g}")
OUT["fe"] = fe_results

# ----------------------------------------------------------------------------
# 4. Distributed-lag (cumulative / medium-run elasticity) for VMT
# ----------------------------------------------------------------------------
dlp = panel.copy().sort_values(["state","t_index"])
K = 12
for k in range(1, K+1):
    dlp[f"lnP_l{k}"] = dlp.groupby("state")["lnP"].shift(k)
dlp = dlp.dropna(subset=[f"lnP_l{k}" for k in range(1,K+1)])
md = pd.get_dummies(dlp["moy"], prefix="m", drop_first=True).astype(float)
lag_cols = ["lnP"] + [f"lnP_l{k}" for k in range(1,K+1)]
Xdl = pd.concat([dlp[lag_cols + ["trend","urate","lninc"]].reset_index(drop=True),
                 md.reset_index(drop=True)], axis=1)
Xdl = sm.add_constant(Xdl)
dlp2 = dlp.reset_index(drop=True)
dlp2["sidx"] = dlp2.groupby("state").cumcount()
midx = dlp2.set_index(["state","sidx"])
Xdl.index = midx.index
res_dl = PanelOLS(midx["lnVMT"], Xdl, entity_effects=True,
                  drop_absorbed=True, check_rank=False).fit(
                  cov_type="clustered", cluster_entity=True)
cum = np.cumsum([res_dl.params[c] for c in lag_cols])
cum_se = []  # cumulative SE via linear combination
cov = res_dl.cov
for h in range(len(lag_cols)):
    sel = lag_cols[:h+1]
    v = cov.loc[sel, sel].values
    cum_se.append(float(np.sqrt(np.ones(h+1) @ v @ np.ones(h+1))))
OUT["dl_cumulative"] = [float(x) for x in cum]
OUT["dl_cum_se"] = cum_se
OUT["dl_longrun"] = float(cum[-1]); OUT["dl_longrun_se"]=cum_se[-1]
print(f"[DL] VMT short-run (h=0) elasticity = {cum[0]:+.3f}")
print(f"[DL] VMT cumulative (h={K}) elasticity = {cum[-1]:+.3f} (se {cum_se[-1]:.3f})")

# ----------------------------------------------------------------------------
# 5. Count model: Negative Binomial for alcohol-impaired crashes, offset=lnVMT
#    plus overdispersion test (Poisson vs NB)
# ----------------------------------------------------------------------------
cm = panel.copy()
cm["offset"] = np.log(cm["VMT"])
state_d = pd.get_dummies(cm["state"], prefix="s", drop_first=True).astype(float)
moy_d = pd.get_dummies(cm["moy"], prefix="m", drop_first=True).astype(float)
Xc = pd.concat([cm[["lnP","alcohol_pc","trend","urate"]].reset_index(drop=True),
                moy_d.reset_index(drop=True), state_d.reset_index(drop=True)], axis=1)
Xc = sm.add_constant(Xc).astype(float)
y_alc = cm["alc_crashes"].astype(float).values

pois = sm.GLM(y_alc, Xc, family=sm.families.Poisson(), offset=cm["offset"].values).fit()
nb   = sm.GLM(y_alc, Xc, family=sm.families.NegativeBinomial(alpha=0.35),
              offset=cm["offset"].values).fit()
# overdispersion: pearson chi2 / df from Poisson
od = float(pois.pearson_chi2 / pois.df_resid)
OUT["nb_alc"] = dict(beta_lnP=float(nb.params["lnP"]),
                     se_lnP=float(nb.bse["lnP"]),
                     beta_alcohol=float(nb.params["alcohol_pc"]),
                     se_alcohol=float(nb.bse["alcohol_pc"]),
                     poisson_dispersion=od,
                     n=int(len(y_alc)))
print(f"[NB] alcohol crashes: lnP coef={nb.params['lnP']:+.3f} (se {nb.bse['lnP']:.3f}); "
      f"alcohol_pc coef={nb.params['alcohol_pc']:+.3f}; Poisson dispersion={od:.2f}")

# speeding NB for completeness
y_spd = cm["speed_fat"].astype(float).values
nb_s = sm.GLM(y_spd, Xc, family=sm.families.NegativeBinomial(alpha=0.25),
              offset=cm["offset"].values).fit()
OUT["nb_speed"] = dict(beta_lnP=float(nb_s.params["lnP"]), se_lnP=float(nb_s.bse["lnP"]))
print(f"[NB] speeding fatalities: lnP coef={nb_s.params['lnP']:+.3f} (se {nb_s.bse['lnP']:.3f})")

# ----------------------------------------------------------------------------
# 6. Threshold / "certain point" regression on total fatalities
#    Hansen-style grid search over candidate price thresholds; report slope
#    below vs above and the bootstrap-style F for threshold existence.
# ----------------------------------------------------------------------------
th = panel.copy()
# within-state + seasonal demeaning of ln_fatal and lnP-level pieces via FE OLS residualization
md2 = pd.get_dummies(th["moy"], prefix="m", drop_first=True).astype(float)
sd2 = pd.get_dummies(th["state"], prefix="s", drop_first=True).astype(float)
base_X = pd.concat([th[["trend","urate"]].reset_index(drop=True),
                    md2.reset_index(drop=True), sd2.reset_index(drop=True)], axis=1)
base_X = sm.add_constant(base_X).astype(float)

grid = np.round(np.arange(2.8, 4.81, 0.10), 2)
ssr_list = []
for g in grid:
    low = np.log(np.minimum(th["P"].values, g))
    high = np.log(np.maximum(th["P"].values/g, 1.0))
    Xg = pd.concat([base_X.reset_index(drop=True),
                    pd.DataFrame({"low":low,"high":high})], axis=1).astype(float)
    m = sm.OLS(th["ln_fatal"].values, Xg).fit()
    ssr_list.append(m.ssr)
ssr_arr = np.array(ssr_list)
thr_hat = float(grid[np.argmin(ssr_arr)])
# linear (no-threshold) restricted model SSR
lin = sm.OLS(th["ln_fatal"].values,
             pd.concat([base_X.reset_index(drop=True),
                        pd.DataFrame({"lnP":th["lnP"].values})],axis=1).astype(float)).fit()
ssr0 = lin.ssr
ssr1 = ssr_arr.min()
n = len(th); fstat = (ssr0-ssr1)/(ssr1/(n - base_X.shape[1]-2))
# slopes at estimated threshold
low = np.log(np.minimum(th["P"].values, thr_hat))
high = np.log(np.maximum(th["P"].values/thr_hat, 1.0))
Xstar = pd.concat([base_X.reset_index(drop=True),
                   pd.DataFrame({"low":low,"high":high})],axis=1).astype(float)
mstar = sm.OLS(th["ln_fatal"].values, Xstar).fit(cov_type="HC1")
OUT["threshold"] = dict(threshold_hat=thr_hat,
                        slope_below=float(mstar.params["low"]),
                        se_below=float(mstar.bse["low"]),
                        slope_above=float(mstar.params["high"]),
                        se_above=float(mstar.bse["high"]),
                        f_existence=float(fstat))
print(f"[THR] estimated threshold = ${thr_hat:.2f}; slope below={mstar.params['low']:+.3f}, "
      f"slope above={mstar.params['high']:+.3f}; F(existence)={fstat:.1f}")

# ----------------------------------------------------------------------------
# 7. National time series: unit roots, cointegration, ECM (price vs fatalities)
# ----------------------------------------------------------------------------
nat = panel.groupby("period").agg(VMT=("VMT","sum"),
                                  fatalities=("fatalities","sum")).reset_index()
nat["lnP"] = lnP_nat            # national price aligned to panel months
nat["lnFat"] = np.log(nat["fatalities"])
nat["lnVMT"] = np.log(nat["VMT"])
# seasonally adjust fatalities & VMT by removing month means (for cleaner TS demo)
for c in ["lnFat","lnVMT","lnP"]:
    moy = np.array([p.month for p in nat["period"]])
    nat[c+"_sa"] = nat[c] - pd.Series(nat[c].values).groupby(moy).transform("mean").values \
                          + nat[c].mean()

def ur(series, regression="c"):
    a = adfuller(series, autolag="AIC", regression=regression)
    try:
        k = kpss(series, regression=("ct" if regression=="ct" else "c"), nlags="auto")
    except Exception:
        k = (np.nan, np.nan)
    return dict(adf_stat=float(a[0]), adf_p=float(a[1]),
                kpss_stat=float(k[0]), kpss_p=float(k[1]))

# Driving (lnVMT) and price (lnP) are the natural long-run pair. lnVMT trends, so
# test it around a deterministic trend; lnP is tested around a constant.
OUT["ur_lnP"]    = ur(nat["lnP_sa"].values, regression="c")
OUT["ur_lnVMT"]  = ur(nat["lnVMT_sa"].values, regression="ct")
OUT["ur_lnFat"]  = ur(nat["lnFat_sa"].values, regression="c")
OUT["ur_dlnP"]   = ur(np.diff(nat["lnP_sa"].values), regression="c")
OUT["ur_dlnVMT"] = ur(np.diff(nat["lnVMT_sa"].values), regression="c")
print(f"[UR] lnP   ADF p={OUT['ur_lnP']['adf_p']:.3f}; d.lnP ADF p={OUT['ur_dlnP']['adf_p']:.3f}")
print(f"[UR] lnVMT ADF p={OUT['ur_lnVMT']['adf_p']:.3f}; d.lnVMT ADF p={OUT['ur_dlnVMT']['adf_p']:.3f}")

# Engle-Granger on the driving-demand relationship (allow trend in cointegrating reg)
eg = coint(nat["lnVMT_sa"].values, nat["lnP_sa"].values, trend="ct")
OUT["eg"] = dict(stat=float(eg[0]), p=float(eg[1]))
# Johansen with linear trend in the data (det_order=1)
joh = coint_johansen(nat[["lnVMT_sa","lnP_sa"]].values, det_order=1, k_ar_diff=2)
OUT["johansen"] = dict(trace0=float(joh.lr1[0]), crit0_5=float(joh.cvt[0,1]),
                       trace1=float(joh.lr1[1]), crit1_5=float(joh.cvt[1,1]))
print(f"[COINT] Engle-Granger p={eg[1]:.3f}; Johansen trace r=0 "
      f"{joh.lr1[0]:.1f} vs 5% {joh.cvt[0,1]:.1f}")

# ECM: long-run driving-demand regression (with trend) then residual-correction
trend_lin = np.arange(len(nat))/12.0
lr = sm.OLS(nat["lnVMT_sa"].values,
            sm.add_constant(np.column_stack([trend_lin, nat["lnP_sa"].values]))).fit()
ect = lr.resid
ecm_df = pd.DataFrame({
    "dVMT": np.diff(nat["lnVMT_sa"].values),
    "dP":   np.diff(nat["lnP_sa"].values),
    "ect_l1": ect[:-1]
}).dropna()
ecm = sm.OLS(ecm_df["dVMT"], sm.add_constant(ecm_df[["dP","ect_l1"]])).fit(
    cov_type="HAC", cov_kwds={"maxlags":6})
OUT["ecm"] = dict(longrun_beta=float(lr.params[2]),
                  speed_adj=float(ecm.params["ect_l1"]),
                  speed_se=float(ecm.bse["ect_l1"]),
                  shortrun_dP=float(ecm.params["dP"]))
print(f"[ECM] long-run beta={lr.params[2]:+.3f}; speed of adjustment="
      f"{ecm.params['ect_l1']:+.3f} (se {ecm.bse['ect_l1']:.3f})")

# ----------------------------------------------------------------------------
# 8. Diagnostic battery on a representative national OLS
#    lnFat ~ lnP + trend + seasonal
# ----------------------------------------------------------------------------
moy = np.array([p.month for p in nat["period"]])
md3 = pd.get_dummies(moy, prefix="m", drop_first=True).astype(float)
Xd = pd.concat([pd.DataFrame({"lnP":nat["lnP"].values,
                              "trend":np.arange(len(nat))/12.0}),
                md3.reset_index(drop=True)], axis=1)
Xd = sm.add_constant(Xd).astype(float)
ols = sm.OLS(nat["lnFat"].values, Xd).fit()
bp = het_breuschpagan(ols.resid, ols.model.exog)
bg = acorr_breusch_godfrey(ols, nlags=12)
dw = durbin_watson(ols.resid)
reset = linear_reset(ols, power=2, use_f=True)
# VIF on the non-dummy regressors
vif_X = sm.add_constant(pd.DataFrame({"lnP":nat["lnP"].values,
                                      "trend":np.arange(len(nat))/12.0,
                                      "urate":panel.groupby('period')['urate'].mean().values}))
vifs = {vif_X.columns[i]: float(variance_inflation_factor(vif_X.values, i))
        for i in range(1, vif_X.shape[1])}
OUT["diagnostics"] = dict(
    bp_stat=float(bp[0]), bp_p=float(bp[1]),
    bg_stat=float(bg[0]), bg_p=float(bg[1]),
    dw=float(dw),
    reset_F=float(reset.fvalue), reset_p=float(reset.pvalue),
    vif=vifs,
)
print(f"[DIAG] BP p={bp[1]:.3f}; BG(12) p={bg[1]:.3g}; DW={dw:.2f}; "
      f"RESET p={reset.pvalue:.3f}; VIF(lnP)={vifs['lnP']:.2f}")

# Hausman FE vs RE on the price coefficient (single-coef version)
re_md = add_seasonal(pidx)
Xre = sm.add_constant(pd.concat([pidx[["lnP","trend","urate","lninc"]], re_md], axis=1))
fe_r = PanelOLS(pidx["lnVMT"], Xre, entity_effects=True, drop_absorbed=True,
                check_rank=False).fit(cov_type="clustered", cluster_entity=True)
re_r = RandomEffects(pidx["lnVMT"], Xre).fit(cov_type="clustered", cluster_entity=True)
bfe, bre = fe_r.params["lnP"], re_r.params["lnP"]
vfe, vre = fe_r.std_errors["lnP"]**2, re_r.std_errors["lnP"]**2
haus = float((bfe-bre)**2 / abs(vfe - vre)) if abs(vfe-vre)>0 else np.nan
OUT["hausman"] = dict(beta_fe=float(bfe), beta_re=float(bre), stat=haus)
print(f"[HAUSMAN] b_FE={bfe:+.3f}, b_RE={bre:+.3f}, chi2(1)~{haus:.2f}")

# ----------------------------------------------------------------------------
# FIGURES
# ----------------------------------------------------------------------------
F = FIG_DIR

# Fig 1 — gas price history (real + nominal) with annotations
fig, ax = plt.subplots(figsize=(9.2, 4.5))
x = price_df.index.to_timestamp()
ax.plot(x, price_df["nominal"], color=NAVY, lw=1.8, label="Nominal $/gal")
ax.plot(x, price_df["real"], color=BURG, lw=1.4, ls="--", label="Real (2024 $)")
ax.set_ylabel("Regular gasoline, $/gallon")
ax.set_title("U.S. retail gasoline prices, 2005–2026", loc="left")
ann = [("2008-07",4.11,"2008 spike\n~$4.11"),
       ("2020-04",1.84,"COVID low\n~$1.84"),
       ("2022-06",5.01,"Record\n~$5.01 (Jun 2022)"),
       ("2026-05",4.56,"2026 shock\n~$4.56")]
for d,v,txt in ann:
    xt = pd.Period(d,"M").to_timestamp()
    ax.scatter([xt],[v], color=GOLD, zorder=5, s=28)
    ax.annotate(txt, (xt,v), textcoords="offset points", xytext=(6,8),
                fontsize=8.5, color="#333")
ax.axhline(4.0, color=GREY, lw=0.8, ls=":")
ax.text(x[5], 4.05, "$4.00 reference", fontsize=8, color=GREY)
ax.legend(frameon=False, loc="upper left", fontsize=9)
ax.set_ylim(1, 6)
fig.text(0.01,-0.02,"Source: constructed from EIA/AAA reported annual averages and episode peaks/lows. "
         "Monthly path interpolated for illustration.", fontsize=7.3, color="#666")
fig.tight_layout(); fig.savefig(f"{F}/fig1_price_history.png", bbox_inches="tight"); plt.close()

# Fig 2 — elasticity landscape (REAL published estimates)
fig, ax = plt.subplots(figsize=(9.2, 4.8))
items = [
 ("VMT / driving, short run", -0.075, (-0.10,-0.05), "Hughes-Knittel-Sperling; lit. range"),
 ("VMT / driving, medium run (CA shock)", -0.22, (-0.25,-0.15), "Gillingham 2014; Knittel-Sandler -0.15"),
 ("VMT / driving, long run", -0.40, (-0.60,-0.20), "literature range"),
 ("Total traffic fatalities", -0.22, (-0.23,-0.20), "Grabowski-Morrisey 2004; Ahangari 2014"),
 ("Drunk-driving crashes (frequency)", -0.27, (-0.34,-0.20), "Chi et al. 2011 (sign/strength)"),
 ("Avg. freeway speed (per $1)", 0.07, (0.04,0.10), "Burger-Kaffine 2009 (rises via decongestion)"),
]
ypos = np.arange(len(items))[::-1]
for (lab,pt,(lo,hi),src),yy in zip(items, ypos):
    col = TEAL if pt<0 else BURG
    ax.plot([lo,hi],[yy,yy], color=col, lw=3, alpha=.5, solid_capstyle="round")
    ax.scatter([pt],[yy], color=col, s=46, zorder=5)
    ax.text(hi+0.02 if pt<0 else hi+0.02, yy, f"  {src}", va="center", fontsize=7.6, color="#555")
ax.axvline(0, color="#222", lw=0.9)
ax.set_yticks(ypos); ax.set_yticklabels([i[0] for i in items], fontsize=9)
ax.set_xlabel("Elasticity with respect to the gasoline price")
ax.set_title("What the published evidence finds", loc="left")
ax.set_xlim(-0.75, 0.55)
fig.text(0.01,-0.03,"Point estimates and approximate ranges from the cited studies. Negative = the "
         "outcome falls when gas prices rise; speed rises because roads de-congest.", fontsize=7.3, color="#666")
fig.tight_layout(); fig.savefig(f"{F}/fig2_elasticity_landscape.png", bbox_inches="tight"); plt.close()

# Fig 3 — stylized facts: real gas price vs total US traffic fatalities (REAL annual)
fat_annual = {2005:43510,2006:42708,2007:41259,2008:37423,2009:33883,2010:32999,
              2011:32479,2012:33782,2013:32894,2014:32744,2015:35484,2016:37806,
              2017:37473,2018:36835,2019:36355,2020:38824,2021:42939,2022:42721,
              2023:40901,2024:39345}
yrs = sorted(fat_annual)
real_annual = [float(real.loc[pd.Period(f"{y}-07","M")]) for y in yrs]
fig, ax1 = plt.subplots(figsize=(9.2,4.5))
ax1.bar(yrs, [fat_annual[y] for y in yrs], color="#cfd8e3", width=0.7, label="Traffic fatalities (left)")
ax1.set_ylabel("U.S. traffic fatalities", color=NAVY)
ax1.set_ylim(25000,46000)
ax2 = ax1.twinx()
ax2.plot(yrs, real_annual, color=BURG, lw=2.2, marker="o", ms=4, label="Real gas price (right)")
ax2.set_ylabel("Real gas price, 2024 $/gal", color=BURG)
ax2.grid(False)
ax1.set_title("Stylized facts: fatalities and the real gas price move together at major turning points",
              loc="left", fontsize=12)
for y,txt in [(2008,"2008\nspike"),(2020,"COVID"),(2022,"2022\nrecord")]:
    ax1.annotate(txt,(y, fat_annual[y]+600), fontsize=8, ha="center", color="#444")
fig.text(0.01,-0.02,"Sources: fatalities — NHTSA/FARS (2024 figure is the NHTSA early estimate); price — EIA, deflated. "
         "Co-movement is descriptive, not causal.", fontsize=7.3, color="#666")
fig.tight_layout(); fig.savefig(f"{F}/fig3_stylized_facts.png", bbox_inches="tight"); plt.close()

# Fig 4 — threshold fit (ILLUSTRATIVE)
fig, ax = plt.subplots(figsize=(8.6,4.4))
# partial-residual scatter: ln_fatal residualized on controls vs price level
resid_y = sm.OLS(th["ln_fatal"].values, base_X).fit().resid
pl = th["P"].values
order = np.argsort(pl)
ax.scatter(pl, resid_y, s=4, alpha=0.06, color=NAVY)
xx = np.linspace(pl.min(), pl.max(), 200)
yhat = (mstar.params["low"]*np.log(np.minimum(xx,thr_hat))
        + mstar.params["high"]*np.log(np.maximum(xx/thr_hat,1.0)))
yhat = yhat - yhat.mean() + resid_y.mean()
ax.plot(xx, yhat, color=BURG, lw=2.4, label="Piecewise (threshold) fit")
ax.axvline(thr_hat, color=GOLD, lw=1.6, ls="--", label=f"Estimated threshold ≈ ${thr_hat:.2f}")
ax.set_xlabel("Gasoline price, $/gallon"); ax.set_ylabel("Fatalities (residualized, logs)")
ax.set_title("Is there a 'certain point'? Estimated threshold in the price–fatality relationship",
             loc="left", fontsize=11.5)
ax.legend(frameon=False, fontsize=9)
fig.text(0.01,-0.03,"ILLUSTRATIVE — calibrated synthetic panel. Demonstrates the Hansen-style threshold "
         "search; not an empirical estimate.", fontsize=7.3, color="#a33")
fig.tight_layout(); fig.savefig(f"{F}/fig4_threshold.png", bbox_inches="tight"); plt.close()

# Fig 5 — cumulative distributed-lag elasticity (ILLUSTRATIVE)
fig, ax = plt.subplots(figsize=(8.6,4.4))
h = np.arange(0,K+1)
cum_arr = np.array(cum); se_arr=np.array(cum_se)
ax.plot(h, cum_arr, color=NAVY, lw=2.2, marker="o", ms=4)
ax.fill_between(h, cum_arr-1.96*se_arr, cum_arr+1.96*se_arr, color=NAVY, alpha=0.15)
ax.axhline(0, color="#222", lw=0.8)
ax.set_xlabel("Months since price change"); ax.set_ylabel("Cumulative VMT elasticity")
ax.set_title("Driving responds gradually: cumulative elasticity of miles to the gas price",
             loc="left", fontsize=11.5)
ax.text(K-0.2, cum_arr[-1], f"  ~{cum_arr[-1]:.2f} by {K} months", va="center", fontsize=8.5, color=NAVY)
fig.text(0.01,-0.03,"ILLUSTRATIVE — calibrated synthetic panel. Shape mirrors the short-run vs medium-run "
         "gap documented in the literature.", fontsize=7.3, color="#a33")
fig.tight_layout(); fig.savefig(f"{F}/fig5_distributed_lag.png", bbox_inches="tight"); plt.close()

# Fig 6 — speeding decomposition (SCHEMATIC / ILLUSTRATIVE)
fig, ax = plt.subplots(figsize=(8.6,4.0))
chans = ["Exposure channel\n(fewer miles driven)","Decongestion channel\n(emptier roads, higher speeds)","Net effect on\nspeeding fatalities"]
vals = [-0.12, 0.08, -0.05]
cols = [TEAL, BURG, NAVY]
bars = ax.bar(chans, vals, color=cols, width=0.55)
ax.axhline(0, color="#222", lw=0.9)
for b,v in zip(bars,vals):
    ax.text(b.get_x()+b.get_width()/2, v + (0.006 if v>=0 else -0.012),
            f"{v:+.2f}", ha="center", fontsize=10, fontweight="bold",
            color="#222")
ax.set_ylabel("Contribution to elasticity")
ax.set_title("Why speeding is ambiguous: two channels nearly cancel", loc="left", fontsize=11.5)
fig.text(0.01,-0.04,"SCHEMATIC / ILLUSTRATIVE. The individual incentive to slow down is weak in the data "
         "(Burger-Kaffine; Frondel-Vance); aggregate speeds rise as roads empty.", fontsize=7.3, color="#a33")
fig.tight_layout(); fig.savefig(f"{F}/fig6_speeding_decomp.png", bbox_inches="tight"); plt.close()

# ----------------------------------------------------------------------------
# Equation images (matplotlib mathtext)
# ----------------------------------------------------------------------------
def eq_png(tex, name, h=0.9, w=8.8, fs=15):
    fig = plt.figure(figsize=(w,h)); fig.patch.set_alpha(0)
    fig.text(0.01, 0.5, tex, fontsize=fs, va="center", ha="left", color="#111")
    fig.savefig(f"{F}/{name}.png", bbox_inches="tight", transparent=True, dpi=200)
    plt.close()

eq_png(r"$\ln Y_{st}=\beta\,\ln P_{st}+\alpha_s+\tau_{m(t)}+\delta\,t+X_{st}^{\prime}\gamma+\varepsilon_{st}$",
       "eq_panel")
eq_png(r"$\ln Y_{st}=\alpha_s+\tau_{m(t)}+\sum_{k=0}^{K}\beta_k\,\ln P_{s,t-k}+X_{st}^{\prime}\gamma+\varepsilon_{st},"
       r"\qquad \beta^{LR}=\sum_{k=0}^{K}\beta_k$", "eq_dl", w=9.4)
eq_png(r"$Y_{st}\sim\mathrm{NegBin}(\mu_{st},\theta),\quad "
       r"\ln\mu_{st}=\beta\,\ln P_{st}+\eta\,A_{st}+\alpha_s+\tau_{m(t)}+\ln(\mathrm{VMT}_{st})$",
       "eq_nb", w=9.6)
eq_png(r"$\ln Y_{st}=\alpha_s+\tau_{m(t)}+\beta_1\min(\ln P_{st},\ln\gamma)+\beta_2\max(\ln P_{st}-\ln\gamma,0)+\varepsilon_{st}$",
       "eq_threshold", w=9.8)
eq_png(r"$\Delta \ln Y_{t}=\phi\,\Delta \ln P_{t}+\lambda\,(\ln Y_{t-1}-\beta\,\ln P_{t-1})+u_t,\quad \lambda<0$",
       "eq_ecm", w=9.2)

print("\nFigures and equation images written to", F)
with open(os.path.join(OUT_DIR, "results.json"),"w") as f:
    json.dump(OUT, f, indent=2)
print("Results JSON written.")

# ----------------------------------------------------------------------------
# 11. EXPORT the synthetic panel to Stata (.dta) and CSV so the do-file runs
#     end-to-end. Column names match gas_driving_analysis.do exactly.
#     This is SYNTHETIC, calibrated data for demonstration, not official data.
# ----------------------------------------------------------------------------
exp = panel.copy()
yr = exp["period"].apply(lambda p: p.year).astype(int)
mo = exp["period"].apply(lambda p: p.month).astype(int)

# national CPI-U path (monthly), anchored ~190.7 (Jan 2005) -> ~314 (Dec 2024)
uniq = list(pd.PeriodIndex(sorted(panel["period"].unique())))
n_per = len(uniq)
cpi_path = {p: 190.7 * (314.0/190.7) ** (i/(n_per-1)) for i, p in enumerate(uniq)}
cpi_col = exp["period"].map(cpi_path).astype(float)

out_df = pd.DataFrame({
    "state":    exp["state"].astype(str),
    "ym":       ((yr - 1960) * 12 + (mo - 1)).astype("int32"),   # Stata %tm integer
    "price":    exp["P"].astype(float).round(4),                  # nominal $/gal
    "cpi":      cpi_col.round(3),                                 # CPI-U index
    "vmt":      exp["VMT"].astype(float).round(5),                # miles (synthetic units)
    "fatal":    exp["fatalities"].astype("int32"),
    "spdfatal": exp["speed_fat"].astype("int32"),
    "duifatal": exp["alc_crashes"].astype("int32"),               # alcohol-related crash proxy
    "urate":    exp["urate"].astype(float).round(3),
    "pcinc":    np.exp(exp["lninc"].astype(float)).round(1),      # per-capita income $
    "etohpc":   exp["alcohol_pc"].astype(float).round(4),         # ethanol gal per capita
    "beertax":  exp["beer_tax"].astype(float).round(4),           # $/gal (time-invariant)
    "gdlyear":  exp["gdl_year"].astype("int32"),
})

vlab = {
    "state": "State postal code",
    "ym": "Month (Stata %tm; apply: format ym %tm)",
    "price": "Retail gasoline price, nominal $/gal (SYNTHETIC)",
    "cpi": "CPI-U index for deflation (SYNTHETIC)",
    "vmt": "Vehicle miles traveled, synthetic units (SYNTHETIC)",
    "fatal": "Total traffic fatalities (SYNTHETIC)",
    "spdfatal": "Speeding-related fatalities (SYNTHETIC)",
    "duifatal": "Alcohol-related crash count, DUI proxy (SYNTHETIC)",
    "urate": "Unemployment rate, percent (SYNTHETIC)",
    "pcinc": "Per-capita personal income, $ (SYNTHETIC)",
    "etohpc": "Apparent ethanol gal per capita (SYNTHETIC)",
    "beertax": "Beer excise tax $/gal (SYNTHETIC)",
    "gdlyear": "Graduated-licensing phase-in year (SYNTHETIC)",
}

out_df.to_csv(os.path.join(DATA_DIR, "gas_driving_panel.csv"), index=False)
out_df.to_stata(os.path.join(DATA_DIR, "gas_driving_panel.dta"), write_index=False,
                version=118, variable_labels=vlab,
                data_label="SYNTHETIC calibrated gas-prices/driving panel (illustrative)")
print(f"Exported gas_driving_panel.dta / .csv: {out_df.shape[0]:,} rows x {out_df.shape[1]} vars")
