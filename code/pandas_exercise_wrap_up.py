#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simulated Time Series, Cointegration, VAR and SVAR Illustration
================================================================

Author: Eric Vansteenberghe
Lecture: Quantitative Methods in Finance (QMF)
License: MIT License

Overview
--------
This script provides a complete pedagogical example illustrating:

1. Simulation of two related time series in returns:
       r_x(t) = α r_x(t−1) + ε(t)
       r_y(t) = β₁ r_y(t−1) + β₂ r_x(t−1) + ν(t)

   with levels reconstructed as:
       x(t) = [1 + r_x(t)] x(t−1)
       y(t) = [1 + r_y(t)] y(t−1)

2. Unit root testing (Augmented Dickey–Fuller).
3. Spurious regression in levels.
4. Cointegration testing.
5. VAR estimation on stationary returns.
6. Structural VAR (SVAR) identification and impulse response analysis.

The objective is to connect econometric theory (unit roots,
cointegration, and structural identification) with a fully
reproducible Python implementation using `statsmodels`.

Structure
---------
• Data Generating Process (DGP)
  - Two autoregressive return processes.
  - Cross-dynamic effect from r_x to r_y.
  - Construction of non-stationary levels from stationary returns.

• Stationarity Analysis
  - Augmented Dickey–Fuller tests in levels and differences.
  - Illustration of I(1) behavior in levels.

• Cointegration
  - Engle–Granger cointegration test.
  - Discussion of spurious regression in non-stationary levels.

• VAR Modeling
  - Lag order selection.
  - Estimation of reduced-form VAR on returns.

• Structural VAR (SVAR)
  - Identification through an A-matrix restriction.
  - Lower-triangular contemporaneous structure:
        A = [[1, 0],
             [E, 1]]
  - Interpretation of structural shocks.
  - Impulse Response Functions (IRFs).

Identification Assumption
-------------------------
The lower-triangular A matrix imposes that shocks to r_x may
contemporaneously affect r_y, but not vice versa. This recursive
ordering yields a causal interpretation consistent with standard
macroeconomic SVAR identification schemes.
"""


import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.api import VAR, SVAR


# create two times series of returns
# rx(t) = alpha rx(t-1) + epsilon(t)
# ry(t) = beta_1 ry(t-1)+ beta_2 rx(t-1) + nu(t)
ryt = [0]
beta1i = .3
beta2i = .5
rxt = [0]
alphai = .2
# in level, we need a starting point
yt = [.1]
xt = [.1]
# length of our data
lent = 1000
# loop to create the time series
for i in range(1,lent):
    ryt.append(beta1i * ryt[i-1] +  beta2i * rxt[i-1]  + np.random.default_rng(i+lent).normal(0,.1,1)[0])
    rxt.append((alphai * rxt[i-1] + np.random.default_rng(i).normal(0,.1,1))[0])
    # by definition, y(t) = (1 + ry(t)) * y(t-1)
    yt.append((1+ryt[i])*yt[i-1])
    xt.append((1+rxt[i])*xt[i-1])

# create the time series in level
df = pd.DataFrame(np.matrix([xt,yt]).transpose())
df.columns = ['xt','yt']

# Add days to the index
start_date = "2020-01-01"
end_date = pd.to_datetime(start_date) + pd.DateOffset(days=lent-1)
date_rng = pd.date_range(start=start_date, end=end_date, freq='D')
df.set_index(date_rng, inplace=True)

# plot our time series
df.plot()

# Augmented Dickey-Fuller test: is there a unit root?
# H0: there is a unit root
# the p-value is the second value in the outcome
# the p-value is the probability to wrongly reject H0
print('ADF test on xt', adfuller(df.xt, regression='c'))
adfuller(df.yt, regression='c')
# there could be some time trend
print('ADF test on yt', adfuller(df.yt, regression='ct'))

# (wrong) first regression in levels
print(smf.ols('yt ~ xt',data=df).fit().summary())

# cointegration test
# H0: no cointegration
# the p-value is the second in the outcome list
print('cointegration test', coint(df.xt,df.yt))

#%% Are our time series I(1)?
dx = (df - df.shift(1)) / - df.shift(1)
dx.dropna(inplace=True)

dx.plot()

# ADF test
print('ADF test on rxt', adfuller(dx.xt, regression='c'))
print('ADF test on ryt', adfuller(dx.yt, regression='c'))

#%% Prepare our VAR with the returns

varmodel = VAR(dx, freq='D')

# select the order of the VAR
print(varmodel.select_order(maxlags=15, trend='c').summary())

#%% SVAR

# we impose no short-run impact of ryt change on rxt change
A = np.array([[1, 0], ['E', 1]])
svarmodel = SVAR(dx, svar_type='A', A=A)
res_svar = svarmodel.fit(maxlags=1, trend='c', solver='nm')

# Normality test of residuals
print(res_svar.test_normality()) # residuals seels normal
 
# residuals autocorrelation test
print(res_svar.test_whiteness()) # no issue here either

# SVAR stabile?
print('SVAR is stable?', res_svar.is_stable()) # stable SVAR

# IRF
res_svar.irf().plot()

