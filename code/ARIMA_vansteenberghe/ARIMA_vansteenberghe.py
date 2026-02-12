"""
QMF — Time Series & Financial Returns (Unit Roots, Random Walks, AR/ARIMA)

This script is a self-contained teaching/replication file used in the Quantitative Methods
in Finance (QMF) materials. It illustrates core concepts in empirical time-series analysis
through daily financial returns, with an emphasis on reproducible workflows and transparent
econometric diagnostics.

Main components
---------------
1) Data acquisition:
   - Optionally downloads the S&P 500 level series (ticker: SP500) from FRED via
     `pandas_datareader` (internet=True).
   - Alternatively loads a local CSV (placeholder: CAC 40 file path) when internet=False.

2) Returns and stylized facts:
   - Computes simple daily returns from the index level.
   - Produces optional plots (index, returns, histograms).

3) Stationarity and unit-root testing (ADF):
   - Simulates (i) white noise, (ii) trend-stationary, and (iii) quadratic-trend processes.
   - Runs ADF tests under different deterministic specifications (c, ct, ctt).
   - Shows how to recover the auxiliary regression output (store=True, regresults=True),
     and reproduces the ADF regression manually via OLS for pedagogical purposes.

4) Forecasting benchmarks:
   - Builds a random-walk forecast for returns and compares forecast errors (ME, MAE, RMSE)
     against a simple “mean so far” benchmark.

5) Simple AR/ARIMA modeling:
   - Uses Ljung–Box tests to assess residual serial correlation.
   - Fits AR(1) models using `statsmodels` ARIMA (order=(1,0,0)).
   - Demonstrates the difference between modeling returns directly and fitting ARIMA(p,1,q)
     on the level series.
   - Provides a simple numerical RMSE minimization for an AR(1) predictor (via `scipy.fmin`).

6) Identification tools:
   - Plots ACF/PACF and shows how to extract ACF confidence intervals.

License
-------
MIT

Author
------
Eric Vansteenberghe
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

if ploton:
    ax = df['index'].plot( title='Index')
    fig = ax.get_figure()
    fig.savefig('fig/cac40.pdf')
    plt.close()
    ax = df['r'].plot( title='Return')
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
AR1model_allatonce = ARIMA(df['index'], order=(1,1,0)).fit()
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

