#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QMF – Financial Time Series: ARIMA and GARCH Modelling
=====================================================

This script provides a compact and reproducible workflow for modelling
equity index returns with standard econometric time-series tools. It is
intended for teaching within the Quantitative Methods in Finance (QMF)
framework, with an emphasis on diagnostics and model checking.

Main components
---------------
1. Data acquisition
   - Downloads the S&P 500 index from FRED (via pandas_datareader), or
     alternatively loads local data.
   - Computes daily simple returns (r_t = P_t/P_{t-1} - 1). A log-return
     variant can be used with np.log(P_t).diff().

2. Mean equation specification (ARIMA)
   - Automatic ARIMA(p,d,q) order selection by BIC using pmdarima.auto_arima.
   - Estimation of the selected ARIMA model using statsmodels.
   - Mean diagnostics:
       * Breusch–Godfrey test for residual serial correlation.

3. Testing for conditional heteroskedasticity (ARCH effects)
   - Diagnostics computed on ARIMA residuals:
       (i) Ljung–Box portmanteau test on squared residuals.
       (ii) Engle’s ARCH LM test for several lag orders m.
       (iii) Manual implementation of the LM statistic (T·R^2).
       (iv) Li–McLeod robust portmanteau test (useful under heavy tails).

4. Volatility modelling (ARCH/GARCH family)
   - ARCH(1) fitted on ARIMA residuals (two-step illustration only).
   - Joint estimation on returns using the arch package:
       * AR(1)-GARCH(1,1)
       * AR(1)-EGARCH(1,1)

5. GARCH-type model checking (after estimation)
   - Diagnostics on standardized residuals z_t = eps_t / sigma_t:
       * Ljung–Box tests on z_t (remaining autocorrelation).
       * Engle ARCH LM tests on z_t (remaining ARCH effects).
       * Jarque–Bera normality test on z_t (often rejected for returns).
   - Optional plots for residuals, squared residuals, and conditional volatility.

Notes
-----
- The two-step ARIMA→ARCH illustration is pedagogical. In applications,
  mean and variance equations should be estimated jointly (as in AR(1)-GARCH
  or AR(1)-EGARCH).

Author: Eric Vansteenberghe
"""

import numpy as np
import pandas as pd
import pmdarima as pm
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model  import ARIMA
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from scipy.optimize import fmin
import statsmodels.formula.api as smf # for linear regressions
from statsmodels.stats import diagnostic
import statsmodels.api as sm
from arch.univariate import ARX
from arch.univariate import GARCH
from arch import arch_model
from pandas_datareader import data as pdr
from scipy.stats import chi2, jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

# We set the working directory (useful to chose the folder where to export output files)
os.chdir('/Users/skimeur/Mon Drive/QMF')

# if you want to plot, set ploton to 1
ploton = True
export = False
internet = True

def download_sp500(start="1950-01-01", end=None):
    """
    Download S&P 500 index (SP500) from FRED.
    
    Parameters
    ----------
    start : str
        Start date in 'YYYY-MM-DD' format.
    end : str or None
        End date in 'YYYY-MM-DD' format. If None, uses today's date.
    
    Returns
    -------
    pd.Series
        Time series of the S&P 500 index.
    """
    df = pdr.DataReader("SP500", "fred", start=start, end=end)
    sp500 = df["SP500"].dropna()
    sp500.name = "SP500"
    return sp500


#%% Import CAC 40 index data
if internet:
    df = download_sp500(start="2000-01-01")
    df = df.to_frame()
    df.columns = ['index']
else:
    df = pd.read_csv('data/cac40.csv', index_col=0, header=None)
    df.columns = ['index']
    # index as dates
    df.index = pd.to_datetime(df.index)

# we compute the daily returns
df['r'] = df['index'] / df['index'].shift(1) - 1

# remove NaN
df = df.dropna()

#%% Fit an ARIMA model
y= df.r
# Auto-select (p,d,q) by BIC; keep it simple
auto = pm.auto_arima(
    y,
    seasonal=False,      # ARIMA (not SARIMA)
    stepwise=True,       # fast heuristic search
    information_criterion="bic",
    suppress_warnings=True,
    error_action="ignore"
)

print("Selected order:", auto.order)

# If you want a statsmodels ARIMA result object with that order:
model = ARIMA(y, order=auto.order).fit()
print(model.summary())

# plot the model residuals
if ploton:
    ax = model.resid.plot()
    fig = ax.get_figure()
    if export:
        fig.savefig('fig/ARMA_residuals.pdf')

# we can test the residuals of the model with a Breusch-Godfrey test
diagnostic.acorr_breusch_godfrey(model,nlags=20)
# H0: there is no serial correlation of any order up to nlags

#%% ------------------------------------------------------------
# ARCH diagnostics (before fitting ARCH/GARCH)
# ------------------------------------------------------------
# We test ARCH effects on ARIMA residuals: eps_t = model.resid
# H0 (no ARCH): E[eps_t^2 | F_{t-1}] = constant
# ------------------------------------------------------------

eps = pd.Series(model.resid, index=y.index, name="eps").dropna()
eps2 = eps**2

# ---------- (i) Portmanteau tests on squared residuals (Ljung–Box) ----------
# If eps_t is i.i.d., eps_t^2 should be serially uncorrelated.
lags_lb = [5, 10, 20, 30]

lb = acorr_ljungbox(eps2, lags=lags_lb, return_df=True)  # Q-stat + p-values
print("\nARCH diagnostic (i): Ljung–Box on squared residuals eps^2")
print(lb.rename(columns={"lb_stat": "Q", "lb_pvalue": "pvalue"}))

# ---------- (ii) Engle ARCH LM test (recommended) ----------
# Auxiliary regression: eps_t^2 = a0 + sum_{i=1}^m a_i eps_{t-i}^2 + u_t
# LM = T * R^2  ~ Chi^2(m) under H0
m_list = [1, 2, 5, 10, 20]
print("\nARCH diagnostic (ii): Engle ARCH LM test (het_arch)")
for m in m_list:
    lm_stat, lm_pval, f_stat, f_pval = het_arch(eps, nlags=m)
    print(f"  m={m:>2d}: LM={lm_stat:>8.3f} (p={lm_pval:.4g}) | F={f_stat:>8.3f} (p={f_pval:.4g})")

# ---------- (iii) Manual Engle LM (explicit T*R^2) ----------
# Same logic as above, written out to match lecture notes exactly.
def engle_lm_manual(eps, m=10):
    e2 = pd.Series(eps).dropna().astype(float) ** 2
    e2.name = "e2"

    X = pd.concat([e2.shift(i) for i in range(1, m + 1)], axis=1)
    X.columns = [f"e2_lag{i}" for i in range(1, m + 1)]
    df_aux = pd.concat([e2, X], axis=1).dropna()

    y_aux = df_aux["e2"]
    X_aux = sm.add_constant(df_aux.drop(columns="e2"))
    res_aux = sm.OLS(y_aux, X_aux).fit()

    T = int(res_aux.nobs)
    R2 = float(res_aux.rsquared)
    LM = T * R2
    pval = 1.0 - chi2.cdf(LM, df=m)
    return LM, pval, T, R2

print("\nARCH diagnostic (iii): Manual Engle LM (LM = T*R^2)")
for m in m_list:
    LM, pval, T, R2 = engle_lm_manual(eps, m=m)
    print(f"  m={m:>2d}: T={T:>5d}, R2={R2:>.4f} -> LM={LM:>8.3f}, p={pval:.4g}")

# ---------- (iv) Li–McLeod test (heavy-tailed finite samples) ----------
# For univariate series, the Li–McLeod statistic reduces to:
#   LMc = T^2 * sum_{i=1}^m r_i^2 / (T-i),  where r_i = corr(eps_t^2, eps_{t-i}^2)
# Under H0: LMc ~ Chi^2(m) (univariate case; multivariate uses Chi^2(k^2 m))
def li_mcleod_univariate(eps, m=20):
    e2 = pd.Series(eps).dropna().astype(float) ** 2
    T = len(e2)
    r = [e2.autocorr(lag=i) for i in range(1, m + 1)]
    stat = (T**2) * sum((r[i-1]**2) / (T - i) for i in range(1, m + 1))
    pval = 1.0 - chi2.cdf(stat, df=m)
    return stat, pval

m_lm = 20
LMc, p_LMc = li_mcleod_univariate(eps, m=m_lm)
print("\nARCH diagnostic (iv): Li–McLeod on squared residuals (univariate)")
print(f"  m={m_lm}: LMc={LMc:.3f}, p={p_LMc:.4g}  (Chi^2({m_lm}) reference)")


# in practice:
# 2 parameters were estimated in the ARIMA(2,0,0), hence ddof=2
diagnostic.het_arch(model.resid, ddof=2, nlags=1) # H0: no ARCH

#%% AR(1)-ARCH(1) model of the returns
# ARCH doc: https://arch.readthedocs.io/en/latest/univariate/univariate.html
# model checks not a strong poin here (cf R)

# apply an ARCH(1) on the ARMA(1,1) residuals
# Nota Bene: this is for illustration only!
# You should fit the ARIMA-GARCH model all at once
arch1 = arch_model(model.resid, p=1, q=0)
res = arch1.fit()
print(res.summary())
if ploton:
    res.plot()
    res.conditional_volatility.plot()
    dir(res)
    res.resid.plot()
    df.r.plot()
    # both look the same, we want to plot epsilon, not sigma_t epsilon_t which is bound to be followint R_t
    res.std_resid.plot()
    # or equivalently
    (res.resid / res.conditional_volatility).plot()

diagnostic.het_arch(res.std_resid) # H0: no ARCH

# AR(1)-GARCH(1,1)
garch11 = arch_model(df.r, p=1, q=1)
garch11.mean = ARX(df.r, lags=[1])
resgarch = garch11.fit()
print(resgarch.summary())
# or
garch11 = arch_model(
    y,
    mean="AR",
    lags=1,
    vol="GARCH",
    p=1,
    q=1,
    dist="normal",   # or "t" for heavy tails (common in returns)
    rescale=True     # helps numerical stability when y is small (daily returns)
)

resgarch = garch11.fit(disp="off")
print(resgarch.summary())
if ploton:
    resgarch.plot()
    resgarch.conditional_volatility.plot()
    dir(resgarch)
    resgarch.resid.plot()
    df.r.plot()
    # both look the same, we plot the standardized residuals
    resgarch.std_resid.plot()

# H0: no ARCH
print("\nEngle ARCH LM on standardized residuals (should not reject if model is adequate):")
print(diagnostic.het_arch(resgarch.std_resid.dropna(), nlags=10))
# returns: (LM stat, LM pval, F stat, F pval)

#%% AR(1)-EGARCH(1)

# Daily returns
y = df["r"].dropna()

# AR(1) in the mean, EGARCH(1,1) in the variance
egarch = arch_model(
    y,
    mean="AR",     # AR mean
    lags=1,        # AR(1)
    vol="EGARCH",  # EGARCH volatility
    p=1,           # |z_{t-1}| term
    o=0,           # asymmetry term (set to 1 for leverage effect)
    q=1,           # log variance lag
    dist="normal", # or "t" for heavy tails (often better for returns)
    rescale=True
)

resegarch = egarch.fit(disp="off")
print(resegarch.summary())

if ploton:
    resegarch.plot()
    resegarch.conditional_volatility.plot()
    dir(resgarch)
    resegarch.resid.plot()
    df.r.plot()
    # both look the same, we plot the standardized residuals
    resegarch.std_resid.plot()

diagnostic.het_arch(resegarch.std_resid.dropna()) # H0: no ARCH

# Exercise:
# ARMA(p,q)-EGARCH(r)
# use information criterion to select the orders p, j and q of the EGARCH


#%% ------------------------------------------------------------
# GARCH-type model checking (use after resgarch / resegarch are fitted)
# ------------------------------------------------------------

def garch_check(res, lags=(10, 20), name="Model"):
    z = pd.Series(res.std_resid).dropna()         # standardized residuals
    z2 = z**2                                     # squared standardized residuals

    print(f"\n--- Diagnostics: {name} ---")

    # (1) Serial correlation of standardized residuals: H0 no autocorrelation
    print("\nLjung–Box on z_t (standardized residuals):")
    print(acorr_ljungbox(z, lags=list(lags), return_df=True))

    # (2) Remaining ARCH in standardized residuals: H0 no ARCH left
    print("\nEngle ARCH LM on z_t (should NOT reject if volatility is well modelled):")
    for m in lags:
        lm, p_lm, f, p_f = het_arch(z, nlags=m)
        print(f"  nlags={m:>2d}: LM p={p_lm:.4g} | F p={p_f:.4g}")

    # (3) Normality of standardized residuals (optional; often rejected in finance)
    jb_stat, jb_p = jarque_bera(z)
    print("\nJarque–Bera on z_t (normality check, optional):")
    print(f"  JB={jb_stat:.3f}, p={jb_p:.4g}")

    # (4) Quick standardized residual plots (optional)
    if ploton:
        z.plot(title=f"{name}: standardized residuals")
        z2.plot(title=f"{name}: squared standardized residuals")

# Run checks
garch_check(resgarch, lags=(10, 20), name="AR(1)-GARCH(1,1)")
garch_check(resegarch, lags=(10, 20), name="AR(1)-EGARCH(1,1)")
