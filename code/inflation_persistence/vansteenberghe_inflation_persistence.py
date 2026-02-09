#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QMF — Quantitative Methods in Finance
Python: inflation persistence and univariate forecasting (EU HICP, monthly)

Companion code for the lecture notes "Quantitative Methods in Finance",
developed by Eric Vansteenberghe (Université Paris 1 Panthéon-Sorbonne,
Master "Finance, Technology & Data").

This script provides a self-contained empirical workflow to illustrate
inflation persistence in monthly euro-area aggregate data (EU headline HICP
index, 2015 = 100):

1) Data preparation
   - Import EU headline HICP index from a local CSV extract (Eurostat format).
   - Convert the time index to pandas datetime and sort chronologically.

2) Seasonal adjustment (lecture-note level)
   - Perform a multiplicative seasonal decomposition (period = 12).
   - Build a seasonally adjusted index as observed / seasonal.

3) Inflation-rate construction
   - Compute month-on-month growth of the SA index.
   - Annualize the monthly rate (× 12) and express it in percent.

4) Persistence diagnostics (Verbeek-style)
   - ADF unit-root tests with a fixed number of lags to illustrate sensitivity.
   - KPSS stationarity tests with varying Newey–West bandwidth (nlags).
   - ACF and PACF inspection.

5) AR modeling and basic forecast uncertainty
   - Select AR order using information criteria (AIC/BIC) via ar_select_order.
   - Estimate an AR(6) benchmark with AutoReg.
   - Diagnose residual autocorrelation using the Ljung–Box test.
   - Derive 1-step forecast standard error using the AR → MA(∞) representation.

6) Stock & Watson (2007) benchmark idea (adapted)
   - Work with changes in inflation (Δi_t).
   - Choose lag order by AIC and fit an AR(AIC) model on Δi_t.

Usage
-----
- Set `ploton = True` to produce time-series plots and residual diagnostics.
- Update `os.chdir(...)` (or replace with a project-relative path) to match
  your local repository structure.
- Expected input file: data/inflation_data.csv

File: vansteenberghe_inflation_persistence.py
Repository: https://github.com/skimeur/QMF

License: MIT (code). See LICENSE at repository root.
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
import scipy.stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.arima_process import arma2ma
#from statsmodels.tsa.x13 import x13_arima_analysis

# Set 'ploton' to True if you want to generate plots, False if not
ploton = False

# Change the working directory to the path where the data file is located
# Make sure to update this path if the data file is located elsewhere on your system
os.chdir('//Users/skimeur/Mon Drive/QMF/')

#%% Inflation persistence analysis starts here
# Reading the CSV file that contains inflation data
df = pd.read_csv('data/inflation_data.csv')

# -----------------------------------------------------------------------------
# Filter inflation data for the European Union (EU)
# -----------------------------------------------------------------------------
# geo    = 'EU'   → European Union aggregate
# unit   = 'I15'  → index (reference year = 2015)
# coicop = 'CP00' → all items (headline inflation)
# -----------------------------------------------------------------------------

# Keep only the relevant observations and variables
df = df.loc[
    (df["geo"] == "EU") &
    (df["unit"] == "I15") &
    (df["coicop"] == "CP00"),
    ["TIME_PERIOD", "OBS_VALUE"]
]

# Convert the time variable to a datetime index
df["TIME_PERIOD"] = pd.to_datetime(df["TIME_PERIOD"])
df = df.set_index("TIME_PERIOD").sort_index()

# -----------------------------------------------------------------------------
# Optional: plot the inflation index over time
# -----------------------------------------------------------------------------
if ploton:
    df["OBS_VALUE"].plot(
        title="EU headline inflation index (2015 = 100)",
        ylabel="Index value",
        xlabel="Time"
    )


# -----------------------------------------------------------------------------
# Seasonal adjustment (simple approach for lecture notes)
# -----------------------------------------------------------------------------
# We use a monthly decomposition with period = 12.
# For an index series like HICP, a multiplicative decomposition is often sensible.
# Note: `seasonal_decompose` yields separate components (trend/seasonal/resid).
# If you want a seasonally-adjusted series, remove the seasonal component.
# -----------------------------------------------------------------------------

decomp = seasonal_decompose(
    df["OBS_VALUE"],
    model="multiplicative",   # seasonal effects scale with the level
    period=12,
    extrapolate_trend="freq"  # avoids NaNs at the sample ends for the trend
)

# Seasonally adjusted series (SA): observed / seasonal (multiplicative case)
df["OBS_VALUE_SA"] = df["OBS_VALUE"] / decomp.seasonal

# (Optional) Keep components for inspection/teaching
# df["trend"]    = decomp.trend
# df["seasonal"] = decomp.seasonal
# df["resid"]    = decomp.resid

# -----------------------------------------------------------------------------
# Optional cross-check (more advanced / system-dependent)
# -----------------------------------------------------------------------------
# X-13ARIMA-SEATS can be used as a robustness check, but it requires an external
# installation and is overkill for this simple example.
# df["OBS_VALUE_SA_X13"] = x13_arima_analysis(df["OBS_VALUE"], freq="M").seasadj

# -----------------------------------------------------------------------------
# Optional: plot the seasonally adjusted inflation index
# -----------------------------------------------------------------------------
if ploton:
    df["OBS_VALUE_SA"].plot(
        title="EU headline inflation index (seasonally adjusted)",
        ylabel="Index value (2015 = 100)",
        xlabel="Time"
    )


# -----------------------------------------------------------------------------
# Construct annualized inflation from the seasonally adjusted index
# -----------------------------------------------------------------------------
# Monthly inflation is computed as the percentage change in the index.
# We then annualize it by multiplying by 12 and express it in percent.
# -----------------------------------------------------------------------------

df["i"] = (
    df["OBS_VALUE_SA"]
    .pct_change()        # monthly growth rate
    .mul(12 * 100)       # annualization and percent scaling
)

# Note:
# This corresponds to an annualized month-on-month inflation rate.


# Drop any rows containing missing values (NaN) from the DataFrame
df.dropna(inplace=True)

# -----------------------------------------------------------------------------
# Optional: plot the annualized monthly inflation rate
# -----------------------------------------------------------------------------
if ploton:
    df["i"].plot(
        title="EU inflation rate (annualized, seasonally adjusted)",
        ylabel="Percent",
        xlabel="Time"
    )


# Augmented Dickey-Fuller test: is there a unit root?
# H0: there is a unit root
adfuller(df.i, regression='c')

# -----------------------------------------------------------------------------
# Unit root diagnostic "à la Verbeek" (Chapter 8)
# -----------------------------------------------------------------------------
# We run the ADF test with a *fixed* number of lags and show how the conclusion
# can change when we add more lags.
#
# ADF null hypothesis (H0): unit root (non-stationarity).
# Verbeek’s point: for inflation, rejections are often marginal and become less
# likely as more lags are included → inflation is either I(1) or highly persistent I(0).
# -----------------------------------------------------------------------------

def adf_with_lags(x: pd.Series, lags: int, regression: str = "c") -> dict:
    # statsmodels versions differ: adfuller returns 5 or 6 elements depending on store/store_results
    out = adfuller(x.dropna(), regression=regression, autolag=None, maxlag=lags)

    stat = out[0]
    pval = out[1]
    usedlag = out[2]
    nobs = out[3]
    crit = out[4]

    return {
        "lags": usedlag,
        "adf_stat": stat,
        "p_value": pval,
        "crit_5": crit["5%"],
        "crit_10": crit["10%"],
    }

# Example lags (Verbeek reports 2 and 4, then “more lags”)
lags_list = [2, 4, 6, 8, 12]

rows = [adf_with_lags(df["i"], L) for L in lags_list]
adf_table = pd.DataFrame(rows)

print("ADF test with intercept (regression='c') — varying the number of lags")
print(adf_table.to_string(index=False, float_format=lambda z: f"{z: .3f}"))

# Interpretation (Verbeek-style):
# Compare adf_stat to critical values: reject H0 if adf_stat < crit value.
# If results are marginal and sensitive to lags, treat inflation as I(1) or very persistent I(0).


# -----------------------------------------------------------------------------
# KPSS test (complementary, Verbeek-style)
# -----------------------------------------------------------------------------
# KPSS null hypothesis (H0): stationarity.
# Verbeek’s point: for inflation, KPSS conclusions can depend on the bandwidth
# choice in the Newey–West correction → use as a diagnostic, not a verdict.
# -----------------------------------------------------------------------------

def kpss_with_bandwidth(x: pd.Series, regression: str = "c", nlags="auto") -> dict:
    stat, pval, usedlags, crit = kpss(
        x.dropna(), regression=regression, nlags=nlags
    )
    return {
        "nlags": usedlags,
        "kpss_stat": stat,
        "p_value": pval,
        "crit_5": crit["5%"],
        "crit_10": crit["10%"],
        "nlags_choice": nlags,
    }

# Show sensitivity to the bandwidth (number of lags in Newey–West)
# - "auto" uses an automatic rule
# - fixed integers force specific bandwidth choices
bandwidth_list = ["auto", 2, 4, 6, 8, 12]

rows = [kpss_with_bandwidth(df["i"], regression="c", nlags=bw) for bw in bandwidth_list]
kpss_table = pd.DataFrame(rows)

print("\nKPSS test with intercept (regression='c') — varying Newey–West bandwidth (nlags)")
print(kpss_table.to_string(index=False, float_format=lambda z: f"{z: .3f}"))

# Interpretation:
# Reject H0 (stationarity) if kpss_stat > critical value.
# If conclusions shift with nlags, emphasize “high persistence” rather than a binary I(0)/I(1) call.


# ACF: high persistence confirmed
sm.graphics.tsa.plot_acf(
    df["i"],
    lags=25
)

# -----------------------------------------------------------------------------
# Partial Autocorrelation Function (PACF)
# -----------------------------------------------------------------------------
# We use the "ywm" method (Yule–Walker, modified):
#   • based on Yule–Walker equations for autoregressive processes,
#   • commonly used as a default in empirical macroeconomics.
# -----------------------------------------------------------------------------

sm.graphics.tsa.plot_pacf(
    df["i"],
    lags=25,
    method="ywm"
)


# -----------------------------------------------------------------------------
# Automatic AR order selection
# -----------------------------------------------------------------------------
# ar_select_order estimates AR(p) models for p = 0, …, maxlag
# and reports information criteria (AIC, BIC) to guide model choice.

mod = ar_select_order(df["i"], maxlag=12)

# Information criteria for each candidate model
mod.aic
mod.bic

# -----------------------------------------------------------------------------
# Estimate the chosen model: AR(6)
# -----------------------------------------------------------------------------
# The AR(6) specification can be motivated by the PACF
# or by minimizing an information criterion.

ar6_model = AutoReg(df["i"], lags=6).fit()

# -----------------------------------------------------------------------------
# Optional: inspect residuals over time
# -----------------------------------------------------------------------------
# Residuals should look roughly uncorrelated and centered around zero.

if ploton:
    ar6_model.resid.plot(title="AR(6) residuals")

# Ljung-Box test on the model residual: H0: The residuals are independently distributed 
acorr_ljungbox(ar6_model.resid, lags=[10], return_df=True)

# -----------------------------------------------------------------------------
# 1-step forecast standard error via AR -> MA(∞)
# -----------------------------------------------------------------------------
# For an AR model, the forecast error variance at horizon h is:
#   Var(e_{t+h|t}) = σ^2 * sum_{j=0}^{h-1} ψ_j^2
# where ψ_j are MA(∞) coefficients. For h=1, only ψ_0 = 1 matters.
# -----------------------------------------------------------------------------

sigma2 = ar6_model.resid.var(ddof=int(ar6_model.df_model) + 1)  # residual variance

# MA(∞) coefficients ψ_j (ψ_0 = 1)
ar_poly = [1] + list(-ar6_model.params[1:])  # AR polynomial: 1 - φ1 L - ... - φp L^p
psi = arma2ma(ar=ar_poly, ma=[1], lags=50)

h = 1
se_h = np.sqrt(sigma2 * np.sum(psi[:h] ** 2))

y_T_plus_1 = ar6_model.predict(start=len(df), end=len(df)).iloc[0]
forecast_interval_low  = y_T_plus_1 - 1.96 * se_h
forecast_interval_high = y_T_plus_1 + 1.96 * se_h

# -----------------------------------------------------------------------------
# Print block 
# -----------------------------------------------------------------------------
level = 95  # since we used 1.96
print(f"1-step ahead point forecast (t+1): {y_T_plus_1:.4f}")
print(
    f"1-step ahead {level}% PI: "
    f"[{forecast_interval_low:.4f}, {forecast_interval_high:.4f}]  (SE={se_h:.4f})"
)

# -----------------------------------------------------------------------------
# Stock & Watson (2007) AR(AIC) benchmark — adapted to monthly EU inflation
# -----------------------------------------------------------------------------
# In Stock & Watson (2007), the baseline univariate forecast is an AR(AIC) model
# estimated on changes in inflation, allowing for a unit root in inflation levels.
#
# Their direct h-step specification is:
#   π^{(h)}_{t+h} - π_t = μ_h + α_h(B) Δπ_t + v^{(h)}_t
# with lag order chosen by AIC (recursively).
#
# Here we keep it simple and implement the key idea:
#   1) work with Δi_t (change in inflation rate)
#   2) pick p by AIC using ar_select_order
#   3) fit AR(p) to Δi_t
# -----------------------------------------------------------------------------

# 1) Construct Δi_t (Stock & Watson use Δπ_t in the AR(AIC) benchmark)
df["di"] = df["i"].diff()
df = df.dropna(subset=["di"])

# 2) Select AR order by AIC (as in AR(AIC))
sel = ar_select_order(df["di"], maxlag=12, ic="aic", trend="c")

p_aic = sel.ar_lags[-1] if sel.ar_lags is not None else 0  # selected lag order
print(f"Selected AR order by AIC (Stock & Watson AR(AIC) idea): p = {p_aic}")

# 3) Fit AR(p) on Δi_t
ar_aic_model = AutoReg(df["di"], lags=p_aic, trend="c").fit()
print(ar_aic_model.summary())

# Optional: residual plot
if ploton:
    ar_aic_model.resid.plot(title=f"AR(AIC) residuals on Δi_t (p={p_aic})")



#%% Import the data on French population again as in part 1
# Import French population data
df = pd.read_csv('data/Valeurs.csv', sep=';', encoding='latin1', skiprows=[0,1,2], header=None, index_col=False)
df = df.iloc[::-1]
df.columns = ['Year','Month','Population']
df.index = pd.to_datetime((df.Year*100+df.Month).apply(str),format='%Y%m')
df = df.drop(['Year','Month'],axis=1)
df = df.replace({' ': ''}, regex=True) 
df = df.astype(float)
df = df / 1000
dx = (df/df.shift(1)-1).dropna()

if ploton:
    dx.plot(title='Population change')

#%% From the average monthly French population change, 
# we want to append to your DataFrame some projections until 2020, 
# plot this projection.

# compute the average monthly change
avgchg = dx.mean()[0]

# create a new df with index of dates we want
dates3 = pd.date_range('1994-01', '2020-01', freq='MS') 
dfproj = pd.DataFrame(index=dates3)
dfproj = dfproj.join(df)

for i in range(0,len(dfproj)-1):
    if np.isnan(dfproj.iloc[i+1,0]):
        dfproj.iloc[i+1] = dfproj.iloc[i] * (1 + avgchg)

if ploton:
    ax = dfproj.plot(title='French population in million projected up to 2020')
    fig = ax.get_figure()
    #fig.savefig('frenchpopproj.pdf')

dfproj_change = (dfproj/dfproj.shift(1))-1
if ploton:
    dfproj_change.plot()

#%% Confidence interval

# ADF test
adfuller(dx.dropna(), regression='c')

# we esitmate the basic model
reg = smf.ols('Population ~ Population.shift(1)', data=dx).fit()
reg.summary()  



# we first create a dfest with the estimated value
# we use copy, deep copy as we don't want dfest to be "linked" with df anymore
dfest = df.copy(deep=True)
for i in range(1,len(dfest)):
    dfest.iloc[i] = df.iloc[i-1]* (1 + avgchg)

dfcompare = pd.concat([df,dfest],axis=1)
if ploton:
    dfcompare.plot()
    dfest.plot()
    df.plot()

#%% Normally distributed error terms
# we usually assume that our error terms are normally distributed
if ploton:
    reg.resid.hist(bins=50)

# check that the mean of our error should be around 0
reg.resid.mean()

# we compute the standard error of the error term and take 5% confidence inteval
stde = reg.resid.std()
# this is the standard error with a forecast at horizon one

# discuss which standard error we should consider for the confidence interval

# normal two sided distribution
z = scipy.stats.norm.ppf(0.975)

if ploton:
    # we want to plot the normal distribution
    resx = 0.00001  # resolution
    x = np.arange(-0.001, 0.001, resx)
    # normal distribution
    pdf = scipy.stats.norm.pdf(x, reg.resid.mean(), reg.resid.std())
    alpha = 0.025  # confidence level
    LeftThres = -scipy.stats.norm.ppf(1-alpha, reg.resid.mean(), reg.resid.std())
    RightThres = scipy.stats.norm.ppf(1-alpha, reg.resid.mean(), reg.resid.std())
    plt.figure(num=1, figsize=(11, 6))
    plt.plot(x, pdf, 'b', label="Normally distributed errors")
    #plt.hold(True)
    plt.axis("tight")
    # Vertical lines
    plt.plot([LeftThres, LeftThres], [0, 250], c='r')
    plt.plot([RightThres, RightThres], [0, 250], c='r')
    plt.xlim([-0.001, 0.001])
    plt.ylim([0, 1800])
    plt.legend(loc="best")
    plt.xlabel("Errors")
    plt.ylabel("Probability of occurence")
    plt.title("Error distribution and confidence at 5%")
    #plt.show()
    #plt.savefig('fig/errors_conf.pdf')

#%% Compute the confidence interval
# at the 95% confidence interval


# what if we take the value from the t distribution:
# tparam = scipy.stats.t.ppf(0.975,len(dx)-2)


#%% Exercise: stick to the monthly seasonality, 
# compute a monthly average population change and use it to project the population up to 2020
