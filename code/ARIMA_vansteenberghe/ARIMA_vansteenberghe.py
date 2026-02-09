#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2024

@author: Eric Vansteenberghe
Quantitative Methods in Finance
Models and Forecast reminder: ARIMA, Garch, VAR
# https://arch.readthedocs.io/en/latest/univariate/introduction.html
"""

import numpy as np
import pandas as pd
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

# We set the working directory (useful to chose the folder where to export output files)
os.chdir('/Users/skimeur/Mon Drive/QMF')

# if you want to plot, set ploton to 1
ploton = False

#%% Import CAC 40 index data
df = pd.read_csv('data/cac40.csv', index_col=0, header=None)
df.columns = ['CAC']
# index as dates
df.index = pd.to_datetime(df.index)

# we compute the daily returns
df['r'] = df['CAC'] / df['CAC'].shift(1) - 1

# remove NaN
df = df.dropna()

if ploton:
    ax = df['CAC'].plot( title='CAC 40 index')
    fig = ax.get_figure()
    fig.savefig('fig/cac40.pdf')
    plt.close()
    ax = df['r'].plot( title='CAC 40 return')
    fig = ax.get_figure()
    fig.savefig('fig/rcac40.pdf')
    plt.close()
    
#%% Histograms
if ploton:
    fig = plt.figure()
    plt.hist(df['r'], 150, facecolor='g', alpha=0.75)
    fig.savefig('fig/hist_rcac40.pdf')
    
    # you might want to norm your data
    fig = plt.figure()
    plt.hist(df['r'], 150, facecolor='g', alpha=0.75, density=True)
    # once it is normed, the integral will sum to one, demonstration:
    fig = plt.figure()
    plt.hist(df['r'], 150, facecolor='g', alpha=0.75, density=True , cumulative=True)

#%% Withe Noise

wn = pd.DataFrame(df.r.mean() + np.random.normal(0, (df.r-df.r.mean()).std(), len(df)))
wn.columns = ['Withe Noise']


if ploton:
    ax = wn['Withe Noise'].iloc[:300,].plot(title = 'White Noise with the mean and SD of the CAC 40 daily returns')
    fig = ax.get_figure()
    fig.savefig('fig/whitenoise.pdf')

# unit root test, how to get the regression output?
# inspired from https://stackoverflow.com/questions/38516846/regression-method-used-in-statsmodels-adfuller
adfstat, pvalue, critvalues, resstore = adfuller(wn['Withe Noise'], regression='c', store=True, regresults=True)
dir(resstore)
print(resstore.resols.summary())


# manual test to investigate the significance of the trend component
# we put Delta Y, Y t-1 and Delta Y t-1
dfadf = pd.concat([wn.diff(1),wn.shift(1),wn.diff(1).shift(1)],axis = 1)
dfadf.columns = ['DY','Y1','DY1']
dfadf = dfadf.dropna(how="any", axis=0)
dfadf.reset_index(inplace=True, drop=True)
dfadf['trend'] = dfadf.index + 1

# we perform the OLS:
resultadf = smf.ols('DY ~  trend + Y1 + DY1',data = dfadf).fit()
resultadf.summary()

del dfadf, resultadf

#%% Trend stationary model

wn['trend stationary'] = wn['Withe Noise'] + 0.001 * wn.index

if ploton:
    ax = wn['trend stationary'].iloc[:300,].plot(title = 'Trend stationary model with the mean and SD of the CAC 40 daily returns')
    fig = ax.get_figure()
    fig.savefig('fig/trendstationary.pdf')

# unit root test
adfuller(wn['trend stationary'],regression='c')
# you need to take the trend into account in your test:
adfuller(wn['trend stationary'],regression='ct')

# manual test to investigate the significance of the trend component
# we put Delta Y, Y t-1 and Delta Y t-1
dfadf = pd.concat([wn['trend stationary'].diff(1),wn['trend stationary'].shift(1),wn['trend stationary'].diff(1).shift(1)],axis = 1)
dfadf.columns = ['DY','Y1','DY1']
dfadf = dfadf.dropna(how="any", axis = 0)
dfadf.reset_index(inplace = True,drop = True)
dfadf['trend'] = dfadf.index + 1

# we perform the OLS:
resultadf = smf.ols('DY ~  trend + Y1 + DY1',data = dfadf).fit()

resultadf.summary()

# Removing the trend
# first difference
if ploton:
    wn['trend stationary'].diff(1).plot()
    
del resultadf, dfadf

#%% Trend and quadratic trend model
    
wn['quadratic trend'] = wn['Withe Noise'] + 0.001 * wn.index + + 0.0001 * np.array(wn.index)**2


if ploton == 1:
    ax = wn['quadratic trend'].iloc[:100,].plot(title = 'Quadratic trend stationary model with the mean and SD of the CAC 40 daily returns')
    fig = ax.get_figure()
    fig.savefig('fig/quadtrendstationary.pdf')

# unit root test
# what happens it you "forget" the quadratic trend? your conclusion are modified
adfuller(wn['quadratic trend'],regression='c')
adfuller(wn['quadratic trend'],regression='ct')
adfuller(wn['quadratic trend'],regression='ctt')

del wn

#%% Random Walk Model

# we "forecast" our CAC 40 as a random walk
rw = pd.DataFrame(df.r.shift(1))
rw.columns = ['Random Walk']
rw.iloc[0,0] = 0

# unit root tests
adfuller(rw['Random Walk'],regression='c')

# mean forecasting error of our model
(rw['Random Walk'] - df.r).mean()

# mean absolute forecast error
(rw['Random Walk'] - df.r).abs().mean()

# root mean squared forecast error
np.sqrt(((rw['Random Walk'] - df.r)**2).mean())

# could compare it with just taking the mean observed so far, not forecasting then
meansofar = pd.DataFrame(df.r) * 0
for t in range(len(df)):
    meansofar.iloc[t,0] = df.iloc[:t,1].mean()
np.sqrt(((meansofar.r.fillna(0) - df.r)**2).mean())

if np.sqrt(((rw['Random Walk'] - df.r)**2).mean()) > np.sqrt(((meansofar.r.fillna(0) - df.r)**2).mean()):
    print('we do not do better than just taking the mean observed so far')

del rw


#%% Autoregressive Model for the CAC 40 index returns

# test for serial correlation
# with the Ljung-Box test. H0: no serial correlation
diagnostic.acorr_ljungbox(df.r, lags=[1], return_df=True)
# we do not reject H0 => weak case for a simple AR(1) model of the CAC daily returns

# nevertheless fit an AR(1) model to the CAC 40 returns in two lines
AR1model = ARIMA(df.r, order=(1,0,0)).fit()
AR1model.summary() 
# indeed, weak case for an AR(1) model

# we try the equivalent ARIMA(1,1,0) on the original time series
# but is it equivalent (all at once versus sequential)?
AR1model_allatonce = ARIMA(df.CAC, order=(1,1,0)).fit()
AR1model_allatonce.summary() 

# My advice: if you have the raw time series, fit an ARIMA (or a SARIMA) directly

del AR1model, meansofar, adfstat, pvalue, critvalues, resstore, t


# import another return time series
df = pd.read_csv('data/GB00B3WK5475.csv', index_col=0).dropna()
df = pd.read_csv('data/PTBCP0AM0015.csv', index_col=0).dropna()

df.columns = ['r']
df.index = pd.to_datetime(df.index)
if ploton:
    df.r.plot(legend=False)

# test for serial correlation
# with the Ljung-Box test. H0: no serial correlation
diagnostic.acorr_ljungbox(df.r, lags=[1], return_df=True)
# we reject H0 

# fit an AR(1) model to the returns in two lines
AR1model = ARIMA(df.r, order=(1,0,0)).fit()
AR1model.summary() # case for an AR(1) model

# we fit an AR(1) model and start with a certain value of phi
c = df.r.mean()
theta = 1
df['rAR'] =  c + theta * df.r.shift(1)
df['rAR'] = df['rAR'].fillna(0)

# minimize the root mean square error
def RMSE(x):
    df['rAR'] =  x[0] + x[1] * df.r.shift(1)
    return np.sqrt(((df.rAR - df.r)**2).mean())

# or with the fminbound function in python
xopt = fmin(RMSE, [0, 1]) 

df['rAR'] = xopt[0] + xopt[1] * df.r.shift(1)
df['rAR'] = df['rAR'].fillna(0)

# unit root test
adfuller(df['rAR'],regression='c') 

# root mean square one-day forecast error
np.sqrt(((df['rAR'] - df.r)**2).mean())

meansofar = pd.DataFrame(df.r) * 0
for t in range(len(df)):
    meansofar.iloc[t,0] = df.iloc[:t,0].mean()

if np.sqrt(((df['rAR'] - df.r)**2).mean()) > np.sqrt(((meansofar.r.fillna(0) - df.r)**2).mean()):
    print('we do not do better than just taking the mean observed so far')
else:
    print('our AR(1) model is better than just taking the mean observed so far')



#%% ACF  and PAC

if ploton:
    plt.figure()
    plot_acf(df.r, lags=5) # MA order ?
    plt.ylim(-.1,.1)
    plt.show()
    plt.figure()
    plot_pacf(df.r, lags=5) # AR order ?
    plt.ylim(-.1,.1)
    plt.show()
# further info on how to choose the ARMA parameters: https://machinelearningmastery.com/grid-search-arima-hyperparameters-with-python/

# display confidence interval
# inspired by: https://stackoverflow.com/questions/62210345/statsmodels-acf-confidence-interval-doesnt-match-python
acf, confidence_interval = sm.tsa.acf(df.r,nlags=df.r.shape[0]-1,alpha=0.05,fft=False)
confidence_interval

#%% ARIMA model to the returns

# inspired by the webpage: https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/

# fit model to the returns
model = ARIMA(df.r, order=(1,0,1)).fit()
# for a model without the constant: 
#model_fit = model.fit(disp=0,trend = 'nc')
print(model.summary())

# based on the ACF and PACF, it was tempting to fit an ARMA(2,2)
# what do you obtain when you fit this?
model22 = ARIMA(df.r, order=(2,0,2)).fit()
# for a model without the constant: 
#model_fit = model.fit(disp=0,trend = 'nc')
print(model22.summary())

# to know the output of the model
dir(model)

model33 = ARIMA(df.r, order=(3,0,3)).fit()

# plot the model residuals
if ploton:
    ax = model33.resid.plot()
    fig = ax.get_figure()
    fig.savefig('fig/ARMA_residuals.pdf')

# we can test the residuals of the model with a Breusch-Godfrey test
diagnostic.acorr_breusch_godfrey(model33,nlags=20)
# H0: there is no serial correlation of any order up to nlags

# manually compute the ARMA
model.params

cstt = model.params[0]
phi = model.params[1]
theta = model.params[2]
df['rARMA'] = np.nan
df.rARMA.iloc[0] = 0
for i in range(1,len(df)):
    df.rARMA.iloc[i] = cstt + phi * df.rARMA.iloc[i-1] + theta * (df.rARMA.iloc[i-1] - df.r.iloc[i-1])

# unit root test
adfuller(df.rARMA,regression='c') 

if np.sqrt(((df['rARMA'] - df.r)**2).mean()) > np.sqrt(((meansofar.r.fillna(0) - df.r)**2).mean()):
    print('our ARMA model do not do better than just taking the mean observed so far')
else:
    if np.sqrt(((df['rARMA'] - df.r)**2).mean()) > np.sqrt(((df['rAR'] - df.r)**2).mean()):
        print('our ARMA(1,1) is better than our AR(1)')
    else: 
        print('our AR(1) model beats our ARMA(1,1)')

# plot residual errors
residuals = pd.DataFrame(model.resid)
diagnostic.het_arch(residuals) # H0: no ARCH
diagnostic.het_arch(model33.resid) # H0: no ARCH
if ploton:
    residuals.plot(legend=False)
    residuals.hist(bins=50)
    residuals.plot(kind='kde')
    print(residuals.describe())
# our result seems rather unbiased

del c, cstt, i, meansofar, phi, residuals, t, theta, xopt, AR1model

#%% AR(1)-ARCH(1) model of the returns
# ARCH doc: https://arch.readthedocs.io/en/latest/univariate/univariate.html
# model checks not a strong poin here (cf R)

# apply an ARCH(1) on the ARMA(1,1) residuals
# Nota Bene: this is for illustration only!
# You should fit the ARIMA-GARCH model all at once
arch1 = arch_model(model33.resid, p=1, q=0)
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
