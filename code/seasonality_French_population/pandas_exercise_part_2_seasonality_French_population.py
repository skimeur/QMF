#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
french_population_seasonality_sarima.py

Quantitative Methods in Finance (QMF) — Python Companion Code
Topic: Seasonality, unit roots, and SARIMA modelling on French population data

Author: Eric Vansteenberghe
Affiliation: Banque de France & Université Paris 1 Panthéon-Sorbonne
Year: 2026

Purpose
---------------------------
Pedagogical, end-to-end illustration of how to diagnose and treat seasonality
in a macroeconomic time series using Python and pandas, with a focus on:
(i) unit-root testing under seasonality, (ii) seasonal dummies and regression-
based tests, (iii) seasonal differencing and decomposition, and (iv) joint
estimation of integration and seasonality via SARIMA.

The example relies on monthly French population data and is designed as a
*beginner-level* but methodologically rigorous exercise for students in
quantitative finance and applied econometrics.

Main steps
----------
1. Data handling with pandas:
   - Import and cleaning of monthly French population data
   - Indexing by dates and sample truncation (pre-Covid focus)

2. Growth-rate construction and stationarity testing:
   - Level vs growth rate
   - Augmented Dickey–Fuller tests with different deterministic components
   - Illustration of the pitfalls of unit-root tests in the presence of seasonality

3. Regression-based seasonality tests:
   - Monthly dummy regressions
   - Interpretation of month-specific effects (September as reference)

4. Seasonal adjustment approaches:
   - Seasonal differencing (lag 12)
   - STL decomposition (trend / seasonality / residual)
   - Classical additive seasonal decomposition
   - Comparison via ADF tests on residual components

5. SARIMA modelling (statsmodels):
   - Grid-search logic (commented, computationally intensive)
   - Selected SARIMA specification
   - Residual diagnostics (ACF/PACF, Ljung–Box)
   - Forecasting growth rates around the Covid-19 period

6. Back-transformation to levels:
   - From forecasted growth rates to projected population levels
   - Construction of confidence intervals in levels

Data & paths
------------
- Expected working directory:
  /Users/skimeur/Mon Drive/QMF/
- Input:
  data/valeurs_mensuelles.csv
- Output figures (optional, controlled by `ploton`):
  fig/frenchpopprojectionSARIMA.pdf

Packages
--------
numpy, pandas, matplotlib, statsmodels (OLS, STL, ADF, SARIMAX),
itertools

Notes
-----
- The script deliberately contrasts *sequential* treatments of seasonality
  (dummies, differencing, decomposition) with *joint* treatment via SARIMA.
- For research-grade work, model selection, diagnostics, and inference should
  be extended (e.g. alternative lag selection, structural breaks, multivariate
  extensions).

Reference
---------
Vansteenberghe, E. (2026). *Quantitative Methods in Finance*.
arXiv preprint arXiv:2601.12896.
"""


import os
import itertools
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
#from pmdarima import auto_arima
import statsmodels.formula.api as smf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import seasonal_decompose

# to plot, set ploton to ploton to 1
ploton = False

# change the working directory
os.chdir('//Users/skimeur/Mon Drive/QMF/')



#%% More granular data set, seasonality and SARIMA
# Import French population data
df = pd.read_csv('data/valeurs_mensuelles.csv', sep=';', encoding='latin1', skiprows=[0,1,2,3], usecols=[0,1], header=None, index_col=False)
# name the columns
df.columns = ['date','pop']
# convert dates to dates
df.date = pd.to_datetime(df.date)
# I want the index as the date
df.index = df.date
# then I can drop the date column
del df['date']
# flip the data set
df = df.iloc[::-1]

# plot on our period of interest, around the Covid-19
if ploton:
    df.loc[(df.index>pd.to_datetime('2019-12'))&(df.index<pd.to_datetime('2021-01')),'pop'].plot(label='Actual Population Level', color='blue')

# keep original data set
dforig = df.copy(deep=True)
# working with dates before the Covid-19 lock-down in france
df = df.loc[df.index<pd.to_datetime('2020-03-01'),:]
# plot the date
if ploton:
    df.plot()

#%% Working with the growth rate
dx = (df - df.shift(1)) / df.shift(1)
dxorign = (dforig - dforig.shift(1)) / dforig.shift(1)
if ploton:
    dx.plot()

#%% Not considering seasonality, Unit root test, is our time series I(1)?
adfuller(df, regression='ct')
adfuller(df, regression='c')
adfuller(df, regression='n')

# We do the test before removing seasonality, which can be problematic
adfuller(dx.dropna(), regression='c')
adfuller(dx.dropna(), regression='n')

#%% Testing September Growth Rate: Regression with monthly dummy
# Adding month dummies
dx['month'] = dx.index.month
dx_dummies = pd.get_dummies(dx['month'], drop_first=True, prefix='m')

# Merging the dummies with the original dataframe
dx_with_dummies = pd.concat([dx, dx_dummies], axis=1)

# Linear Regression Model
model = smf.ols('pop ~ m_9', data=dx_with_dummies).fit()
print(model.summary())

# The coefficient of m_9 indicates the difference in growth rate for September
# T-test and p-value for m_9 shows if this difference is statistically significant

# Linear Regression Model, considering each month as a categorical variable
# Here, I use September as my reference, this is to test if there is a month where it is statistically greater on average than September
model_categorical = smf.ols('pop ~ C(month, Treatment(reference=9))', data=dx_with_dummies).fit()
print(model_categorical .summary())
# There seem to be no month with a statistically significant greater growth rate than september

del dx['month']

#%% seasonal differencing
# Differencing to remove MONTHLY seasonality
dx_diff = dx - dx.shift(12)  # Assuming monthly data with yearly seasonality
dx_diff = dx_diff.dropna()

if ploton:
    dx_diff.plot()

# ADF Test on the growht rate
print("ADF Test on growth rate:")
print(adfuller(dx.dropna(), regression='c'))

# ADF Test after seasonal differencing on the growht rate
print("ADF Test after Differencing to remove seasonality:")
print(adfuller(dx_diff.dropna(), regression='c'))

#%% STL Decomposition
stl = STL(dx['pop'].dropna(), seasonal=13)
result = stl.fit()
seasonal, trend, resid = result.seasonal, result.trend, result.resid

# Plotting the components
if ploton:
    plt.figure(figsize=(12, 8))
    plt.subplot(411)
    plt.plot(df['pop'], label='Original')
    plt.legend(loc='upper left')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='upper left')
    plt.subplot(413)
    plt.plot(seasonal, label='Seasonality')
    plt.legend(loc='upper left')
    plt.subplot(414)
    plt.plot(resid, label='Residuals')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

# time series without the seasonality
dx_non_seasonal = trend + resid
if ploton:
    dx_non_seasonal.plot()
    
# ADF Test after removing the seasonality with STL
print("ADF Test after STL to remove seasonality and trend:")
print(adfuller(resid.dropna(), regression='c'))
# the residuals are I(0)

#%% Considering seasonality with a simpler seasonality decomposition (alternative, simpler method)
# Seasonality Decomposition
# Here, the model "imposes" that the seasonality is very stable over time, which might not be that realistic as habits can change
decomposition = seasonal_decompose(dx.dropna(), model='additive', period=12)  # Assuming monthly data with yearly seasonality
if ploton:
    fig = decomposition.plot()
    plt.show()
  
# ADF Test after removing the seasonality with STL
print("ADF Test after simple additive decomposition to remove seasonality and trend:")
print(adfuller(decomposition.resid.dropna(), regression='c'))
# the residuals are I(0)


#%% SARIMA

# this is time consuming:
"""
# Define the p, d, q parameters to take any value between 0 and 2
p = d = q = range(0, 3)

# Define seasonal parameters
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

# Grid Search
best_aic = float("inf")
best_pdq = None
best_seasonal_pdq = None
best_results = None

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = SARIMAX(dx.dropna(), order=param, seasonal_order=param_seasonal, enforce_stationarity=False, enforce_invertibility=False)
            results = mod.fit()
            
            if results.aic < best_aic:
                best_aic = results.aic
                best_pdq = param
                best_seasonal_pdq = param_seasonal
                best_results = results
        except:
            continue

print('SARIMA{}x{} - AIC:{}'.format(best_pdq, best_seasonal_pdq, best_aic))
"""
# Summary of the best model
# SARIMA(0, 1, 0)x(1, 0, 1, 12) 
best_model = SARIMAX(dx.dropna(), order=(0,1,0), seasonal_order=(1,0,1,12), enforce_stationarity=False, enforce_invertibility=False)
best_results = best_model.fit()
print(best_results.summary())
best_results_latex = best_results.summary().as_latex()

# Does this faster method finds the same outcome?
#model = auto_arima(dx.dropna(), seasonal=True, m=12, stepwise=True, trace=True, error_action='ignore', suppress_warnings=True)

print(model.summary())

# Plotting the residuals
residuals = best_results.resid
if ploton:
    plt.figure(figsize=(10, 6))
    plt.plot(residuals)
    plt.title('Residuals from SARIMA Model')
    plt.xlabel('Date')
    plt.ylabel('Residuals')
    plt.show()
    
# Checking for Autocorrelation (ACF and PACF plots)
if ploton:
    fig, ax = plt.subplots(1, 2, figsize=(15, 4))
    sm.graphics.tsa.plot_acf(residuals, lags=40, ax=ax[0])
    sm.graphics.tsa.plot_pacf(residuals, lags=40, ax=ax[1])
    plt.show()
    
# Ljung-Box test
lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
# Null-hypothesis: H0: There is no autocorrelation in the residuals up to lag 10
print(lb_test) # we reject H0, we need to apply an additional model, like an ANN to the residuals

#%% Predict the population growht rates
# Define the number of steps to forecast (from March 2020 to October 2020)
n_steps = 8
# Forecasting the future values
forecast_results = best_results.get_prediction(start=pd.to_datetime('2020-03-01'), end=pd.to_datetime('2020-10-01'), dynamic=True)
forecast = forecast_results.predicted_mean
confidence_intervals = forecast_results.conf_int()

if ploton:
    # Plotting the results
    plt.figure(figsize=(12, 6))
    ax = dxorign.loc[(dxorign.index>pd.to_datetime('2018-01'))&(dxorign.index<pd.to_datetime('2020-11')),'pop'].plot(label='Actual', color='blue')  # Actual data
    
    # Plotting the forecasted values
    forecast.plot(ax=ax, label='Forecast', alpha=0.7, color='red')
    
    # Plotting the confidence intervals
    ax.fill_between(confidence_intervals.index,
                    confidence_intervals.iloc[:, 0],
                    confidence_intervals.iloc[:, 1], color='pink', alpha=0.3)
    
    # Setting the plot labels and legend
    ax.set_xlabel('Date')
    ax.set_ylabel('Population Growth Rate')
    plt.legend()
    plt.title('Population Growth Rate: Actual vs Forecast')
    plt.show()

# Plot in level
# Compute the last "known" population level
last_known_pop = df.loc[pd.to_datetime('2020-02'),'pop']

# Adjust the forecasted growth rates 
adjusted_forecast = 1 + forecast

# Compute the cumulative product of the adjusted forecasted growth rates
cumulative_growth_factor = adjusted_forecast.cumprod()

# Compute the projected population level
projected_population = last_known_pop * cumulative_growth_factor

# Compute confidence intervals for the projected population
# Adjusting the confidence intervals similarly
confidence_intervals_adjusted = 1 + confidence_intervals
confidence_intervals['lower_pop'] = last_known_pop * confidence_intervals_adjusted.iloc[:, 0].cumprod()
confidence_intervals['upper_pop'] = last_known_pop * confidence_intervals_adjusted.iloc[:, 1].cumprod()

# Plotting the results
if ploton:
    plt.figure(figsize=(12, 6))
    ax = dforig.loc[(dforig.index>pd.to_datetime('2017-01'))&(dforig.index<pd.to_datetime('2020-11')),'pop'].plot(label='Actual Population Level', color='blue')
    
    # Plotting the projected population levels
    projected_population.plot(ax=ax, label='Projected Population Level', color='red')
    
    # Plotting the confidence intervals
    ax.fill_between(confidence_intervals.index,
                    confidence_intervals['lower_pop'],
                    confidence_intervals['upper_pop'], color='pink', alpha=0.3)
    
    # Setting the plot labels and legend
    ax.set_xlabel('Date')
    ax.set_ylabel('Population Level (K)')
    plt.legend()
    plt.title('Population Level: Actual vs Projected')
    plt.savefig('fig/frenchpopprojectionSARIMA.pdf')
    plt.show()
