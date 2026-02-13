#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qmf_var_svar_identification.py
QMF 2026 — VAR / SVAR in Python: Lag Selection, Identification, and Impulse Responses

This script accompanies the QMF (Quantitative Methods in Finance) lecture notes section
on multivariate time-series models. It implements a reproducible workflow for:

(1) Data preparation (France)
    - INSEE-based population series (monthly, resampled to quarterly)
    - GDP series (quarterly)
    - Unemployment series (quarterly)
    - Basic checks for seasonality/trend (illustrative regression on month dummies)
    - Construction of growth rates (stationary transformations)

(2) Reduced-form VAR
    - VAR estimation with statsmodels
    - Lag order selection via information criteria (AIC, BIC, HQIC, FPE)
    - Stability checks (roots / eigenvalues)
    - Residual diagnostics (normality, whiteness)
    - Granger causality tests (with caveats)
    - Forecasting and (reduced-form) impulse response functions

(3) Structural VAR (SVAR)
    - Short-run identification via an A-matrix restriction (Cholesky-type ordering)
    - Comparison with an LDL decomposition of the reduced-form residual covariance matrix
    - Orthogonalized impulse responses

(4) Additional applications
    - A textbook VAR example (Bourbonnais, Dunod): stationarity checks, VAR estimation,
      causality tests, and impulse responses; rolling OLS illustration
    - Commodity examples (wheat, rice): unit roots, cointegration check, and suggested
      VAR/Granger/IRF extensions

Usage
-----
- Set `ploton = True` to display and/or save figures (PDFs) in `fig/`.
- Update `os.chdir(...)` to your local project path.
- Note: reading legacy `.XLS` files requires `xlrd>=2.0.1`. Alternatively convert the
  file to `.xlsx` and use openpyxl.

Pedagogical notes
-----------------
- Reduced-form VAR impulse responses without orthogonalization are not structural; for
  causal interpretation, identification assumptions are required (SVAR).
- Granger causality tests are sensitive to detrending, lag selection, and omitted variables.
- For non-stationary but cointegrated variables, consider VECM rather than VAR in differences.

Author: Eric Vansteenberghe
Course: Quantitative Methods in Finance (QMF)
Year: 2026

License
-------
MIT License
"""


import pandas as pd
import os
import numpy as np
import statsmodels.tsa.stattools
from statsmodels.tsa.api import VAR, SVAR
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.linalg as la # for LDL decomposition
import matplotlib.pyplot as plt


# to plot, set ploton to ploton to True
ploton = False

# change the working directory
os.chdir('//Users/skimeur/Mon Drive/QMF/')

#%% Import the data on French population again as in part 1
df = pd.read_csv('data/Valeurs.csv', sep=';', encoding='latin1', skiprows=[0,1,2], header=None, index_col=False)
df = df.iloc[::-1]
df.columns = ['Year','Month','Population']
df.index = pd.to_datetime((df.Year*100+df.Month).apply(str),format='%Y%m')
df = df.drop(columns=['Year','Month'])
df = df.replace({' ': ''}, regex=True) 
df = df.astype(float)
df = df / 1000

#%% GDP data
gdp = pd.read_csv('data/GDP.csv', sep=';', encoding='latin1', skiprows = [0,1])
gdp = gdp.iloc[::-1]
gdp.columns = ['Quarter','GDP'] 
gdp['Year'] = gdp.index
gdp.index = pd.to_datetime(gdp['Year'].astype(str) + ['-Q'] + gdp['Quarter'].astype(str))
gdp = gdp.drop(columns=['Year','Quarter'])
gdp = gdp.replace({' ': ''}, regex=True) 
gdp = gdp.astype(float)

#%% Unemployment data in France
u = pd.read_csv('data/unemployment_france.csv', sep=';', encoding='latin1', skiprows = [0,1])
u = u.iloc[::-1]
u.columns = ['Quarter','u']
u['Year'] = u.index
u.index = pd.to_datetime(u['Year'].astype(str) + ['-Q'] + u['Quarter'].astype(str)) 
u = u.drop(columns=['Year','Quarter'])
u = u.replace({' ': ''}, regex=True) 
u = u.replace({',':'.'}, regex=True) 
u = u.astype(float)

#%% Check for trends and seasonality before concatenated the variables

dx = (df - df.shift(1)) / df.shift(1)
dx['month'] = dx.index.month.astype(str)

seasonalitycehck = smf.ols('Population ~  month',data = dx).fit()
seasonalitycehck.summary() # remove this seasonality first!

gdpx = (gdp - gdp.shift(1)) / gdp.shift(1)
gdpx['month'] = gdpx.index.month.astype(str)

seasonalitycehck_gdp = smf.ols('GDP ~  month',data = gdpx).fit()
seasonalitycehck_gdp.summary()

#%% Concatenate all variables in one data frame
dfall = pd.concat([u,df,gdp],axis=1)
dfall = dfall.dropna()
dfallx = (dfall - dfall.shift(1)) / dfall.shift(1)
dfallx = dfallx.resample('Q').mean()

dfallx.dropna().to_csv('dfallx.csv')

del df, dfall, dx, gdp, gdpx, seasonalitycehck, seasonalitycehck_gdp, u


#%% VAR model

# plot the return data
if ploton:
    dfallx.plot(secondary_y = 'u')

# drop rows with missign values (NA)
dfallx = dfallx.dropna()

# VAR with population and GDP growth rates
dfallx2 = dfallx.loc[:, ['Population','GDP']]
dfallx2 = dfallx2.resample('Q').mean()

# create a VAR
varmodel = VAR(dfallx2, freq='Q')

# select the order of the VAR
print(varmodel.select_order(maxlags=15, trend='c').summary())
# you might want to check with no constant term
# but this is unclear to me when you'd want to do this!!!
#print(varmodel.select_order(maxlags=15, trend='n').summary())
# select lag order 1
result_var = varmodel.fit(maxlags=1)
# with no constant term
#result_var = varmodel.fit(1,trend = 'n')
result_var.summary()

# check what information you can get from the VAR
dir(result_var)

# get the matrix
matrixVAR = result_var.params.iloc[1:,]

# compute the eigenvalues
eigenvaluesVAR = np.linalg.eig(matrixVAR)[0]
sum(eigenvaluesVAR >= 1)
# no eigenvalue greater or equal to 1, our VAR is stable
# then the reduced form VAR presented in equation can be consistently 
# estimated by OLS equation by equation.

# other way to check for the VAR stability
# our VAR is stable if the roots lies outside the unit circle
sum(result_var.roots <= 1) 
# notroots are greater or equal to one, our VAR is stable

# add lagged variables
dfallx2['GDP1'] = dfallx2['GDP'].shift(1)
dfallx2['Population1'] = dfallx2['Population'].shift(1)

# estimation by OLS of the reduced form VAR
resultpop = smf.ols('Population ~ Population1 + GDP1',data = dfallx2).fit()
resultGDP = smf.ols('GDP ~ Population1 + GDP1',data = dfallx2).fit()

del dfallx2['GDP1'], dfallx2['Population1']

resultpop.summary()
resultGDP.summary()
# indeed, as no eigenvalue greater or equal to 1, the coefficients are the same

# plot impulse response functions, no orthogonalization (see SVAR and Cholesky decomposition for more details)
if ploton:
    result_var.irf().plot(orth=False)
    # so these IRF are meaningless! 
    # we need a SVAR decomposition first

# do we have a diagonal Omega?
np.cov(result_var.resid.T)

# Granger causality
print(result_var.test_causality('Population','GDP'))

# Normality test of residuals
print(result_var.test_normality()) # residuals are non-normal

# plot time series autocorrelation functions
result_var.plot_acorr()
 
# residuals autocorrelation test
print(result_var.test_whiteness())

lag_order = result_var.k_ar
result_var.forecast(dfallx2.loc[:,['Population','GDP']].values[-lag_order:], 5)

result_var.plot_forecast(10)



#%% SVAR

# we impose no short-run impact of GDP change on population change
A = np.array([[1, 0], ['E', 1]])

svarmodel = SVAR(dfallx2, svar_type='A', A=A)
res_svar = svarmodel.fit(maxlags=1, trend='c', solver='nm')


# SVAR stable?
res_svar.is_stable()

# Normality test of residuals
print(res_svar.test_normality()) # residuals are non-normal

# plot time series autocorrelation functions
res_svar.plot_acorr()
 
# residuals autocorrelation test
print(res_svar.test_whiteness())

# estimated A matrix by our SVAR   
Asvar = res_svar.A

# compare with the Cholesky decomposition
   
# add lagged variables
dfallx2['GDP1'] = dfallx2['GDP'].shift(1)
dfallx2['Population1'] = dfallx2['Population'].shift(1)

# compute the matrix D from the schock terms
resultpop = smf.ols('Population ~ GDP + Population1 + GDP1',data = dfallx2).fit()
resultGDP = smf.ols('GDP ~ Population + Population1 + GDP1',data = dfallx2).fit()

del dfallx2['GDP1'], dfallx2['Population1']

Dmverif = np.matrix([[resultpop.resid.std()**2,0],[0,resultGDP.resid.std()**2]])

# compute the Omega matrix from the forecast error terms
Omegam = np.cov(result_var.resid.T)
# we do a Cholesky decomposition, a LDL decomposition
Ainv, Dm, p = la.ldl(Omegam)

# check our decomposition
Omegam - Ainv.dot(Dm).dot(Ainv.T)

# We found the matrix A^{-1}
Am = np.linalg.inv(Ainv)

if ploton:
    res_svar.irf().plot(orth=False)
    
# In practice, this decomposition is done for you
# so you can use the irf command with orth=True in the plot

if ploton:
    plt.figure()
    result_var.irf().plot(orth=True)
    plt.savefig('fig/irfVARex2.pdf')
    

#%% Add unemployment to the VAR
    # create a VAR
varmodelx = VAR(dfallx,freq='Q')

# select the order of the VAR
print(varmodelx.select_order(maxlags = 15,trend = 'c'))
# you might want to check with no constant term
#print(varmodel.select_order(maxlags = 15,trend = 'n'))
# select lag order 1
result_varx = varmodelx.fit(1)

result_varx.summary()

dir(result_varx)

# Normality test of residuals
print(result_varx.test_normality()) # residuals are non-normal

# plot time series autocorrelation functions
result_varx.plot_acorr()
 
# residuals autocorrelation test
print(result_varx.test_whiteness())

if ploton:
    result_varx.irf().plot(orth=False)

# add the unemployment and test Granger causality between GDP and unemployment
print(result_varx.test_causality('u','GDP'))

del matrixVAR, eigenvaluesVAR, Ainv, Am, Dm, Dmverif, lag_order, Omegam, p, result_var, resultGDP, resultpop, varmodel


#%% From Regis Bourbonnais book "Econometrie" Dunod

# import the data set, drop missing values, set the dates as index
bour = pd.read_excel('data/C10EX2.XLS')
bour = bour.dropna(axis=0)
bour.index = pd.date_range('2001-01', '2019-01', freq='Q')
bour = bour.drop('Date',axis=1)

#bour.plot()

# ADF test
statsmodels.tsa.stattools.adfuller(bour['Y1'], regression='n')
statsmodels.tsa.stattools.adfuller(bour['Y2'], regression='n')
# both series are stationary

# we define a function to print if the series has unit root or not
def hasUR(df,threshold):
    if statsmodels.tsa.stattools.adfuller(df, regression='n')[1] > threshold:
        print("your series has a unit root")
    else:
        print("Your series doesn't seem to have unit root")
    
# we set a threshold at 5% for the p-value of or ADF test
thresholdi = 0.05
hasUR(bour['Y1'],thresholdi)
hasUR(bour['Y2'],thresholdi)

# apply the VAR model
varmodelb = VAR(bour)

#Lag selection
print(varmodelb.select_order(15)) # we select order 1
# fit a VAR(1)
result_varb = varmodelb.fit(1)
result_varb.summary()

#  check for the VAR stability
# our VAR is stable if the roots lies in the unit circle
sum(result_varb.roots <= 1)
sum(result_varb.roots < 1) # one root is greater than one, our VAR is not stable, no possibility of cointegration


# is Y2 causing Y1, our test says yes
print(result_varb.test_causality('Y1','Y2'))
#print(result_varb.test_causality('Y2','Y1'))

# impulse response funciton plot
if ploton:
    result_varb.irf().plot()

#Let's define a moving OLS ourselves:
def movOLS(df,window):
    Betas = []
    df.columns = ['X','Y']
    for i in range(0,(len(df)-window)):
        resultat = sm.OLS(df[i:(i+window)].X,df[i:(i+window)].Y).fit()
        Betas.append(resultat.params[0])
    return Betas;
    
window = 10
betas = movOLS(bour[['Y2','Y1']],window)
if ploton:
    pd.DataFrame(betas).plot(title = 'evolution of the beta over time')

debut = pd.DataFrame(np.zeros(window))

Betas = pd.concat([debut, pd.DataFrame(betas)], ignore_index=True)
Betas.index = bour.index

bour['Y2_hatbis'] = Betas.multiply(bour['Y1'],axis=0)
bour.columns = ['Y1','Y2','Y2 hat']
if ploton:
    ax = bour.loc[bour.index.year>2003,['Y2 hat','Y2']].plot()
    fig = ax.get_figure()
    fig.savefig('fig/bourbonnais_Yhat.pdf')

#diff=bour['Y2_hat']-bour['Y2_hatbis']
#diff.plot()

del Betas, betas, bour, debut, thresholdi, window


#%% Wheat production and price:
wheatprice = pd.read_csv('data/146908e8-7a8a-40ea-b670-b39daab67a15.csv')
wheatprod = pd.read_csv('data/b24e9b0e-4b97-4acc-90f8-599cc178d434.csv')

wheatprice2 = wheatprice.groupby('Year')['Value'].mean()
wheatprod2 = wheatprod.groupby('Year')['Value'].sum()

wheat = pd.concat([wheatprice2,wheatprod2],axis=1).dropna(axis=0)
wheat.columns = ['price','supply']
if ploton:
    wheat.plot(secondary_y='price',title='Wheat price and supply, yearly')

## equivalence between diff() and using shift(1)
#wheat1stdiff = wheatprod2.diff()
#wheat1stdiffcheck = wheatprod2 - wheatprod2.shift(1)
#
#dfcompare = pd.concat([wheat1stdiff,wheat1stdiffcheck], axis=1)
#dfcompare.plot()

# Correlation
wheat.corr()


#ADF test
statsmodels.tsa.stattools.adfuller(wheat['price'])
statsmodels.tsa.stattools.adfuller(wheat['supply'])
# both time series have a unit root

# we compute the growth rate
wheatx = wheat/wheat.shift(1) - 1
wheatx = wheatx.dropna(axis=0)
#ADF test
statsmodels.tsa.stattools.adfuller(wheatx['price'])
statsmodels.tsa.stattools.adfuller(wheatx['supply'])
# both time series are now stationary

#%% Cointegration test
statsmodels.tsa.stattools.coint(wheat.price,wheat.supply)
# The probability to wrongly reject H0 is too high, we accept it, there is no cointegration
# No need for a VECM


#%% compare supply and price

# is the supply at the next period influenced by the price today? and vice-versa?
wheatx.loc[:,'supply1'] = wheatx.loc[:,'supply'].shift(1)
wheatx.loc[:,'price1'] = wheatx.loc[:,'price'].shift(1)

wheatx.loc[:,['price','supply1']].corr()
wheatx.loc[:,['price1','supply']].corr()

if ploton:
    ax = wheatx.plot(title = '1: means lag at t-1')
    fig = ax.get_figure()
    #fig.savefig('wheatsupply1.PNG')

wheat['price1'] = wheat['price'].shift(1)

if ploton:
    axw = wheat.loc[:,['price','supply']].plot(secondary_y = 'price',title='wheat price and supply in levels')
    figw = axw.get_figure()
    #figw.savefig('wheatsupplylevel.pdf')

if ploton:
    axw = wheat.loc[:,['price1','supply']].plot(secondary_y = 'price1',title='wheat price and supply a time t + 1 in levels')
    figw = axw.get_figure()
    #figw.savefig('wheatsupplylevel1.pdf')


# EXERCISE: apply a VAR and test for Granger causality, then do impulse response functions


#%% Rice production and prices:
riceprod = pd.read_csv('data/5413f61a-51fe-423f-a942-64990af7d5e1.csv')
riceprice = pd.read_csv('data/cc87ba8c-13ed-4ba1-ba6e-f3584f8395bc.csv')

riceprice2 = riceprice.groupby('Year')['Value'].mean()
riceprod2 = riceprod.groupby('Year')['Value'].sum()

rice = pd.concat([riceprice2,riceprod2],axis=1).dropna(axis=0)
rice.columns = ['price','supply']
if ploton:
    rice.plot(secondary_y='price')

rice.corr()

# EXERCISE: apply a VAR and test for Granger causality, then do impulse response functions
