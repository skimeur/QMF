#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QMF – Financial Time Series: ARIMA and GARCH Modelling
=======================================================

This script provides a compact and reproducible workflow for modelling
equity index returns using standard econometric time series techniques.
It is designed for teaching and research purposes within the Quantitative
Methods in Finance (QMF) framework.

Main components
---------------
1. Data acquisition
   - Downloads the S&P 500 index from FRED (via pandas_datareader),
     or alternatively loads local data.
   - Computes daily log-returns (simple returns by default).

2. Mean equation specification
   - Automatic ARIMA(p,d,q) selection using BIC (pmdarima.auto_arima).
   - Estimation of the selected ARIMA model using statsmodels.
   - Residual diagnostics:
       * Breusch–Godfrey test for serial correlation
       * ARCH test for conditional heteroskedasticity

3. Volatility modelling
   - ARCH(1) applied to ARIMA residuals (illustrative two-step approach).
   - Joint estimation of:
       * AR(1)-GARCH(1,1)
       * AR(1)-EGARCH(1,1)
     using the `arch` package.
   - Analysis of conditional volatility and standardized residuals.


Notes
-----
- The two-step ARIMA-ARCH approach is presented for illustration only.
  In empirical applications, mean and variance equations should be
  estimated jointly.
- Set `ploton = True` to generate and export diagnostic figures.
- The working directory should be adapted to the user’s environment.

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

# We set the working directory (useful to chose the folder where to export output files)
os.chdir('/Users/skimeur/Mon Drive/QMF')

# if you want to plot, set ploton to 1
ploton = False
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
    fig.savefig('fig/ARMA_residuals.pdf')

# we can test the residuals of the model with a Breusch-Godfrey test
diagnostic.acorr_breusch_godfrey(model,nlags=20)
# H0: there is no serial correlation of any order up to nlags


diagnostic.het_arch(model.resid) # H0: no ARCH

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
if ploton:
    resgarch.plot()
    resgarch.conditional_volatility.plot()
    dir(resgarch)
    resgarch.resid.plot()
    df.r.plot()
    # both look the same, we plot the standardized residuals
    resgarch.std_resid.plot()


#%% AR(1)-EGARCH(1)

egarch = arch_model(df.r, vol='EGARCH', p=1, o=0, q=1)
egarch.mean = ARX(df.r, lags=[1])
resegarch = egarch.fit()
print(resegarch.summary())
if ploton:
    resegarch.plot()
    resegarch.conditional_volatility.plot()
    dir(resgarch)
    resegarch.resid.plot()
    df.r.plot()
    # both look the same, we plot the standardized residuals
    resegarch.std_resid.plot()

diagnostic.het_arch(resegarch.std_resid) # H0: no ARCH

# Exercise:
# ARMA(p,q)-EGARCH(r)
# use information criterion to select the orders p, j and q of the EGARCH
