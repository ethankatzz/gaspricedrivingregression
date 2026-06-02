clear all
set more off

* set Statas working directory to the repository root before running, e.g.:
*   cd "path/to/gas-prices-driving-analysis"
use "data/gas_driving_panel.dta", clear

encode state, gen(stateid)
gen mdate = ym
format mdate %tm
xtset stateid mdate

gen lprice   = ln(price)
gen lvmt     = ln(vmt)
gen lfatal   = ln(max(fatal,1))
gen lspd     = ln(max(spdfatal,1))
gen ldui     = ln(max(duifatal,1))
gen lpcinc   = ln(pcinc)
gen month    = month(dofm(mdate))
gen year     = year(dofm(mdate))
gen trend    = (year-2005) + (month-1)/12
gen lnexpose = ln(vmt)

xtreg lvmt   lprice i.month trend urate lpcinc, fe vce(cluster stateid)
xtreg lfatal lprice i.month trend urate lpcinc, fe vce(cluster stateid)
xtreg ldui   lprice i.month trend urate lpcinc etohpc, fe vce(cluster stateid)
xtreg lspd   lprice i.month trend urate lpcinc, fe vce(cluster stateid)

quietly xtreg lvmt lprice i.month trend urate lpcinc, fe
estimates store fe_vmt
quietly xtreg lvmt lprice i.month trend urate lpcinc, re
estimates store re_vmt
hausman fe_vmt re_vmt, sigmamore

xtreg lvmt L(0/12).lprice i.month trend urate lpcinc, fe vce(cluster stateid)
cap noisily lincom L0.lprice + L1.lprice + L2.lprice + L3.lprice + L4.lprice + L5.lprice + L6.lprice + L7.lprice + L8.lprice + L9.lprice + L10.lprice + L11.lprice + L12.lprice

nbreg duifatal lprice etohpc i.month trend urate i.stateid, offset(lnexpose)
nbreg spdfatal lprice etohpc i.month trend urate i.stateid, offset(lnexpose)

quietly reg lfatal i.stateid i.month trend urate lpcinc
predict lfatal_resid, resid
cap noisily threshold lfatal_resid, threshvar(price) regionvars(lprice) nthresholds(1)

quietly reg lfatal lprice i.stateid i.month trend urate lpcinc
scalar ssr_lin = e(rss)
scalar bestssr = .
scalar bestthr = .
forvalues g = 280/480 {
    local thr = `g'/100
    capture drop plow phigh
    gen plow  = ln(min(price, `thr'))
    gen phigh = ln(max(price/`thr', 1))
    quietly reg lfatal plow phigh i.stateid i.month trend urate lpcinc
    if e(rss) < bestssr {
        scalar bestssr = e(rss)
        scalar bestthr = `thr'
    }
}
capture drop plow phigh
gen plow  = ln(min(price, bestthr))
gen phigh = ln(max(price/bestthr, 1))
reg lfatal plow phigh i.stateid i.month trend urate lpcinc, vce(cluster stateid)
scalar Fexist = ((ssr_lin - bestssr)/1) / (bestssr/(e(N) - e(rank)))
di "threshold = " bestthr "    F(existence) = " Fexist

gen lprice2 = lprice^2
reg lfatal lprice lprice2 i.stateid i.month trend urate lpcinc, vce(cluster stateid)
mkspline pk1 4 pk2 = price
reg lfatal pk1 pk2 i.stateid i.month trend urate lpcinc, vce(cluster stateid)

preserve
collapse (mean) lprice (sum) vmt fatal, by(mdate)
gen lvmt = ln(vmt)
gen lfatal = ln(fatal)
tsset mdate
gen mm = month(dofm(mdate))
foreach v in lprice lvmt lfatal {
    quietly egen `v'_m = mean(`v'), by(mm)
    quietly summarize `v'
    gen `v'_sa = `v' - `v'_m + r(mean)
}
dfuller lprice_sa, lags(12)
dfgls   lprice_sa, maxlag(12)
pperron lprice_sa, lags(12)
dfuller lvmt_sa, trend lags(12)
dfgls   lvmt_sa, maxlag(12)
pperron lvmt_sa, trend lags(12)
dfuller D.lprice_sa, lags(12)
dfuller D.lvmt_sa, lags(12)
reg lvmt_sa trend lprice_sa
predict ect, resid
dfuller ect, noconstant lags(12)
vecrank lvmt_sa lprice_sa, trend(trend) lags(3)
reg D.lvmt_sa D.lprice_sa L.ect, vce(robust)
newey D.lvmt_sa D.lprice_sa L.ect, lag(6)
restore

preserve
collapse (mean) lprice urate (sum) vmt fatal, by(mdate)
gen lfatal = ln(fatal)
gen mm = month(dofm(mdate))
gen trend = _n/12
tsset mdate
reg lfatal lprice trend i.mm
estat hettest
estat imtest, white
estat bgodfrey, lags(1 6 12)
estat durbinalt
estat dwatson
estat ovtest
estat vif
restore
