#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QMF 2026 — Cointegration, ECM, and Regression Diagnostics (French GDP & Population)

This script accompanies the QMF (Quantitative Methods in Finance) lecture notes section on
non-stationary time series. It provides an end-to-end, pedagogical workflow covering:

1) Data loading and basic cleaning
   - French population (monthly; INSEE export)
   - French GDP (quarterly; CSV prepared from INSEE or class materials)

2) Exploratory analysis before regression
   - Time-series plots with dual y-axis
   - Growth-rate computations
   - Scatter plots and correlation in differences

3) Unit root testing
   - Manual Dickey–Fuller regression illustration (single-lag, no intercept version)
   - Augmented Dickey–Fuller tests via statsmodels with different deterministic components

4) Spurious regression and cointegration
   - Illustration of spurious regression risks with I(1) series
   - Engle–Granger cointegration test (statsmodels.tsa.stattools.coint)

5) Synthetic example
   - Simulation of two I(1) series that are cointegrated by construction
   - Estimation of a cointegrating regression and an Error Correction Model (ECM)

6) Empirical example (interest rates)
   - ECB Statistical Data Warehouse Euribor 1Y vs 3M
   - Cointegration check and simple ECM specification

7) Robustness and diagnostics in regression on stationary transformations
   - OLS on growth rates
   - Outlier detection via dummy regressors + Bonferroni-style threshold
   - Q–Q plots of (Pearson) normalized residuals


Author: Eric Vansteenberghe
Affiliation: Banque de France
License: MIT
Year: 2026
"""


import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools
import statsmodels.formula.api as smf
from scipy.stats import ttest_ind
import scipy.stats

# to plot, set ploton to ploton to 1
ploton = False

# change the working directory
os.chdir('//Users/skimeur/Mon Drive/QMF/')

#%% Import the data on French population again as in part 1
# Load the CSV file 'Valeurs.csv' with a semicolon separator, Latin-1 encoding, skip the first three rows, and set no header or index column
df = pd.read_csv('data/Valeurs.csv', sep=';', encoding='latin1', skiprows=[0,1,2], header=None, index_col=False)

# Reverse the DataFrame rows to make the data appear in chronological order, it was initially reversed
df = df.iloc[::-1]

# Rename columns to 'Year', 'Month', and 'Population' for clarity
df.columns = ['Year', 'Month', 'Population']

# Set the index to a monthly date range starting from January 1994 to October 2016
df.index = pd.date_range('1994-01', '2016-10', freq='M')

# Drop the 'Year' and 'Month' columns as they are no longer needed after setting the date index
df = df.drop(columns=['Year', 'Month'])

# Replace any spaces in the 'Population' column with an empty string to clean the data
df = df.replace({' ': ''}, regex=True)

# Convert the 'Population' column values to floats for numerical operations
df = df.astype(float)

# Scale the 'Population' values down by dividing by 1000
df = df / 1000


#%% GDP data

gdp = pd.read_csv('data/GDP.csv',sep=';',encoding='latin1',skiprows = [0,1])
gdp = gdp.iloc[::-1]
gdp.columns = ['Quarter','GDP']
gdp.index = pd.date_range('1949-01', '2016-09', freq='Q') 
gdp = gdp.drop(columns=['Quarter'])
gdp = gdp.replace({' ': ''}, regex=True) 
gdp = gdp.astype(float)
if ploton:
    gdp.plot()

#%% concatenate both time series into one data frame
both = pd.concat([df,gdp],axis=1)

# drop rows with missing values
both = both.dropna(axis=0)

if ploton:
    ax = both.plot(secondary_y=['Population'])
    fig = ax.get_figure()
    fig.savefig('fig/GDP_pop.pdf')

# compute quarterly changes
both_change = (both-both.shift(1))/both.shift(1)

if ploton:
    ax2 = both_change.plot()
    fig2 = ax2.get_figure()
    fig2.savefig('fig/GDP_pop_change.pdf')

#%% Before the OLS: visulization of the data set

if ploton:
    axall = both_change.plot.scatter(x='GDP', y='Population')
    figall = axall.get_figure()
    figall.savefig('fig/gdppopscatterplot.pdf')
    plt.close()

#%% Correlation between population and GDP growth rates

both_change.corr()

del both, both_change


#%% Unit Root test - Dickey-Fuller test
# on gdp data
dickeyfullerdf = gdp.copy(deep=True)
dickeyfullerdf['Deltagdp'] = (gdp - gdp.shift(1))
dickeyfullerdf['GDP1'] = dickeyfullerdf['GDP'].shift(1)
# OLS of the delta GDP on the GDP
dickeyfuller_reg = smf.ols('Deltagdp ~ GDP1 -1', data = dickeyfullerdf).fit()
dickeyfuller_reg.summary()

# we should compare the t statistics of the GDP coefficient against
# the critical value of the t-distribution at 95%
# the degree of freedom are we estimate one coefficient):
degreeoffreedom = len(dickeyfullerdf.dropna()) - 2
proba = 0.95
scipy.stats.t.ppf(proba, degreeoffreedom)

# 10.41 is above 1.65
# low risk to wrongly reject H0 but
# for simplicity here we decide that rho - 1 is not equal to 0
# we decide that there is no unit root int his time series (be careful as the lenght of this series is limited)

del degreeoffreedom, dickeyfullerdf, proba

#%% Unit Root test - Augmented Dickey-Fuller test
#Before we start regressing variables:
#Test integration order:

#we compute the changes
df_change = (df-df.shift(1)) / df.shift(1)
df_change = df_change.dropna()
gdp_change = (gdp-gdp.shift(1)) / gdp.shift(1)
gdp_change = gdp_change.dropna()

#Population
statsmodels.tsa.stattools.adfuller(df.Population, regression='ct')
statsmodels.tsa.stattools.adfuller(df_change.Population, regression='c')
#GDP
statsmodels.tsa.stattools.adfuller(gdp.GDP, regression='ct')
statsmodels.tsa.stattools.adfuller(gdp_change.GDP, regression='c')

# imposing 'nc' to regression mean that we assume a random walk
# imposing 'c' means you assume a random walk with a drift
# imposing 'ct' would have ment that both series could have been trend stationary, in which cas the trend t should have been added in the regression

del df_change, gdp_change

#%% Concatenate all variables in one data frame
dfall = pd.concat([df,gdp],axis=1)
dfall = dfall.dropna()
dfallx = dfall / dfall.shift(1) - 1


spurreg = smf.ols('GDP ~ Population', data=dfallx).fit()
spurreg.summary()


#%% cointegration test
# H0: no cointegration
statsmodels.tsa.stattools.coint(dfall.GDP, dfall.Population)

#%% Cointegation: building two cointegrated time series

xt = [0]
yt = [0]
# define the beta of the system
beta_coint = 0.3
beta_y = 0.9
# length of our data
lent = 1000
for i in range(1,lent):
    xt.append((xt[i-1] + np.random.normal(0,1,1))[0])
    yt.append((beta_y * yt[i-1] + beta_coint * xt[i] + np.random.normal(0,1,1))[0])

dfcoint = pd.DataFrame(np.matrix([yt,xt]).transpose())
dfcoint.columns = ['yt','xt']

if ploton:
    ax = dfcoint.loc[:,['yt','xt']].plot(title='Two cointegrated time series, with both a unit root')
    fig = ax.get_figure()
    #fig.savefig('fig/cointillustration.pdf')

# Unit root test, H0: there is a unit root
statsmodels.tsa.stattools.adfuller(dfcoint['yt'], regression='n')
statsmodels.tsa.stattools.adfuller(dfcoint['xt'], regression='n')
if ploton:
    dfcoint.diff().plot()
statsmodels.tsa.stattools.adfuller(dfcoint['yt'].diff().dropna(), regression='n')
statsmodels.tsa.stattools.adfuller(dfcoint['xt'].diff().dropna(), regression='n')
# both series are I(1)

# Cointegration test
# H0: no cointegration
statsmodels.tsa.stattools.coint(dfcoint['yt'],dfcoint['xt']) # cointegration

dfcoint.mean()
# as series mean are different than 0 we add a constant in the cointegration regression

# regress one series on the other for the cointegration regression
model_coint = smf.ols('yt ~ xt',data=dfcoint).fit()
model_coint.summary()
dir(model_coint) # we can use "resid" from the model to work on the residuals of this regression
# we check that the cointegrating residuals are I(0)
# Unit root test, H0: there is a unit root
# in fact, as demonstrated in Phillips and Ouliaris (1990), one cannot use the ADF test because of the spurious nature of the regression
#statsmodels.tsa.stattools.adfuller(model_coint.resid) # cointegrating residuals are stationary, hence I(0)

dfcoint['y_predicted'] = model_coint.params[0] + model_coint.params[1] * dfcoint['xt']

if ploton:
    ax = dfcoint.loc[:,['yt','y_predicted']].plot(title='Forces de rappel')
    fig = ax.get_figure()
    #fig.savefig('cointforcesrappel.pdf')


dfcoint['Dxt'] = dfcoint['xt'].diff()
dfcoint['Dyt'] = dfcoint['yt'].diff()
dfcoint['Dxt1'] = dfcoint['Dxt'].shift(1)
dfcoint['Dyt1'] = dfcoint['Dyt'].shift(1)
dfcoint['cointerr'] = model_coint.resid

ecm1 = smf.ols('Dyt ~ cointerr + Dyt1 + Dxt1',data=dfcoint).fit()
ecm2 = smf.ols('Dxt ~ cointerr + Dyt1 + Dxt1',data=dfcoint).fit()

ecm1.summary()
ecm2.summary()

#%% Cointegrated data: an example

#%% Cointegrated variables, exercise: Euribor 1 year and 3 months
# Data sources:
# http://sdw.ecb.europa.eu/quickview.do?SERIES_KEY=143.FM.M.U2.EUR.RT.MM.EURIBOR1YD_.HSTA
# Euribor 1 year
eu1 = pd.read_csv('data/FM.M.U2.EUR.RT.MM.EURIBOR1YD_.HSTA.csv',skiprows=4)
eu1 = eu1.iloc[::-1]
eu1.columns = ['date','Euribor1year']
eu1.index = pd.to_datetime(eu1['date'],format = '%Y%b')
del eu1['date']

# Euribor 3 months
eu3 = pd.read_csv('data/FM.M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA.csv',skiprows=4)
eu3 = eu3.iloc[::-1]
eu3.columns = ['date','Euribor3months']
eu3.index = pd.to_datetime(eu3['date'],format = '%Y%b')
del eu3['date']

dfeuribor = pd.concat([eu1,eu3],axis=1)
dfeuribor['spread'] = dfeuribor['Euribor1year'] - dfeuribor['Euribor3months']
if ploton:
    dfeuribor.plot()


dfeuriborx = (dfeuribor-dfeuribor.shift(1)) / dfeuribor.shift(1)
dfeuriborx = dfeuriborx.dropna()

# Unit root test, H0: there is a unit root
statsmodels.tsa.stattools.adfuller(dfeuribor['Euribor1year'], regression='c')
statsmodels.tsa.stattools.adfuller(dfeuriborx['Euribor1year'], regression='c')
statsmodels.tsa.stattools.adfuller(dfeuribor['Euribor3months'], regression='c')
statsmodels.tsa.stattools.adfuller(dfeuriborx['Euribor3months'], regression='c')
# both series are I(1)

# Cointegration test
# H0: no cointegration
statsmodels.tsa.stattools.coint(dfeuribor['Euribor1year'], dfeuribor['Euribor3months']) # cointegration at the 5% threshold


# show that the residuals are stationary for this regression
modeleuribor = smf.ols('Euribor1year ~ Euribor3months',data=dfeuribor).fit()
dir(modeleuribor)
# in fact, as demonstrated in Phillips and Ouliaris (1990), one cannot use the ADF test because of the spurious nature of the regression
# in our case, both series have drifts, then unit root test statistics follow the DF distributions adjusted for a constant and trend
statsmodels.tsa.stattools.adfuller(modeleuribor.resid,regression="c")
statsmodels.tsa.stattools.adfuller(modeleuribor.resid,regression="ct")

# build an ECM with the Euribor rates
dfeuribord = dfeuribor - dfeuribor.shift(1)
dfeuribord['errors'] = modeleuribor.resid.shift(1)
dfeuribord = dfeuribord.dropna()

ecmeuribor = smf.ols('Euribor1year ~ Euribor3months + errors',data=dfeuribord).fit()
ecmeuribor.summary()

# idea: could introduce half-life of shocks

dfeuribord['LE1'] = dfeuribord.Euribor1year.shift(1)
dfeuribord['LE3'] = dfeuribord.Euribor3months.shift(1)


ecmeuribor2 = smf.ols('Euribor1year ~ Euribor3months + errors + LE1 + LE3',data=dfeuribord).fit()
ecmeuribor2.summary()

del dfeuribor, dfeuriborx, eu1, eu3, dfeuribord

#%% Find cointegrated variables
# Import the data set
pwt = pd.read_csv('data/pwt90.csv',encoding='latin1')

# keep only the data for Italy
pwt = pwt.loc[pwt.country=='Italy']
# keep only the column with the year, cgdpe and cda data
pwt = pwt.loc[:,['year','cgdpe','cda']]
# this can be done in one line:
#pwt = pwt.loc[pwt.country=='Italy',['year','cgdpe','cda']]

# set the year as the index
pwt.index = pwt['year']
del pwt['year']

# compute yearly changes and drop the first line which has only NA
pwtx = (pwt - pwt.shift(1)) / pwt.shift(1)
pwtx = pwtx.dropna()

if ploton:
    pwt.plot()

#pwtx.plot() # there seem to be a drift in the returns

# Unit root test, H0: there is a unit root
statsmodels.tsa.stattools.adfuller(pwt['cda'], regression='ct')
statsmodels.tsa.stattools.adfuller(pwt['cgdpe'], regression='ct')
statsmodels.tsa.stattools.adfuller(pwtx['cda'], regression='c')
statsmodels.tsa.stattools.adfuller(pwtx['cgdpe'], regression='c')
statsmodels.tsa.stattools.adfuller(pwtx['cda'].diff().dropna(), regression='c')
statsmodels.tsa.stattools.adfuller(pwtx['cgdpe'].diff().dropna(), regression='c')
# both series are I(2), an ECM would not be very well suited

# Plot the Italy's values
if ploton:
    ax = pwt.plot(title = "Italy's Expenditure-side real GDP and Real domestic absorption")
    fig = ax.get_figure()
    #fig.savefig('Italy_cointegrated.pdf')

# Do the test for the United Kingdom and the same variables

del pwt, pwtx


#%% OLS
# we can do a linear regression on stationary series, the returns
results = smf.ols('Population ~ GDP',data = dfallx).fit()
dir(results)
results.summary()
results.rsquared - dfallx.loc[:,['Population','GDP']].corr().iloc[0,1]**2


del df, gdp

#%% Visual of the regression

dir(results)
results.params

# we tae the constant and the estimated beta
alpha = results.params[0]
beta = results.params[1]

# plot the data
stepsize = 0.001
x = np.arange(1.1*dfallx['GDP'].min(),1.1*dfallx['GDP'].max(),stepsize)

if ploton:
    plt.scatter(dfallx['GDP'],dfallx['Population'])
    plt.xlabel('French GDP changes')
    plt.ylabel('French Population changes')
    plt.plot(x, alpha+beta*x ,'-')
    #plt.savefig('fig/GDP_Pop_scatter.pdf')


#df=df.resample('Q').mean()

#del alpha, beta, beta2, x, stepsize

#%% Outlier detection and regression coefficients robustness
# we detect a first outlier with the minimum value of GDP growth rate
outlier1 = pd.DataFrame(dfallx.sort_values(by='GDP',ascending=True).iloc[0,:])
# second outlier with the maximum growth rate of population
outlier2 = pd.DataFrame(dfallx.sort_values(by='Population',ascending=False).iloc[0,:])

# create dummy variable for both outlier
dfallx['dummy1'] = 0
dfallx.loc[outlier1.columns,'dummy1'] = 1
dfallx['dummy2'] = 0
dfallx.loc[outlier2.columns,'dummy2'] = 1

# outlier 1 test
results_outlier1 = smf.ols('Population ~ GDP + dummy1',data = dfallx).fit()
results_outlier1.summary()
# outlier 2 test
results_outlier2 = smf.ols('Population ~ GDP + dummy2',data = dfallx).fit()
results_outlier2.summary()
# the second outlier seems to be influential
# use the Bonferroni adjustment for the critical threshold
tstatBonferroni = scipy.stats.t.ppf(1-0.05/(2*len(dfallx)),len(dfallx)-2-1)
# we are still above the critical threshold and we reject H0, our dummy2 coefficient is statistically significantly different from 0

# we take the second outlier out of the data set and regress again
results = smf.ols('Population ~ GDP',data = dfallx).fit()

results_outlier2_out = smf.ols('Population ~ GDP',data = dfallx.loc[~ (dfallx.index == outlier2.columns[0]),:]).fit()

results.summary()
results_outlier2_out.summary()

if ploton:
    plt.scatter(dfallx['GDP'],dfallx['Population'])
    plt.xlabel('French GDP changes')
    plt.ylabel('French Population changes')
    plt.plot(x, alpha+beta*x ,'-')
    plt.plot(x, results_outlier2_out.params[0]+results_outlier2_out.params[1]*x ,'-')

    del axes

del outlier1, outlier2, tstatBonferroni

#%% Q-Q plots

dir(results)
# we use the normalized residuals
if ploton:
    scipy.stats.probplot(results.resid_pearson, dist="norm", plot=plt)
    plt.title("Normal Q-Q plot")
    plt.show()
    
if ploton:
    scipy.stats.probplot(results_outlier1.resid_pearson, dist="norm", plot=plt)
    plt.title("Normal Q-Q plot, outlier 1 taken out")
    plt.show()

if ploton:
    scipy.stats.probplot(results_outlier2_out.resid_pearson, dist="norm", plot=plt)
    plt.title("Normal Q-Q plot, outlier 2 taken out")
    plt.show()

