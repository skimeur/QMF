#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QMF — Quantitative Methods in Finance
Stationarity, Unit Roots, Dickey–Fuller / Augmented Dickey–Fuller, AR processes

This script accompanies Section "Stationarity, Unit Roots"
of the lecture notes *Quantitative Methods in Finance* by Eric Vansteenberghe,
developed over more than ten years of teaching at Université Paris 1 Panthéon-Sorbonne
(Master Finance, Technology & Data).

Pedagogical objectives:
- Introduce the notion of (non-)stationarity in time series
- Illustrate unit-root dynamics and the persistence of shocks
- Implement Dickey–Fuller and Augmented Dickey–Fuller (ADF) unit-root tests
- Compare regression choices in ADF ('n', 'c', 'ct', 'ctt') and interpret outputs
- Demonstrate spurious regressions with independent I(1) processes (Granger–Newbold)
- Build simple AR(2) examples and diagnose stability via eigenvalues / roots
- Provide minimal plotting hooks for visual intuition (toggle with `ploton`)

Main topics covered:
- Random walk / unit root and shock persistence
- Simple returns vs. levels for diagnostics
- ADF test with statsmodels (autolag options and regression specifications)
- Spurious regression: inflated R² and invalid inference under non-stationarity
- AR(2) dynamics, companion-matrix eigenvalues, and unit-root cases
- Brief empirical example (world population, log + trend removal)

Intended audience:
- Economics and finance students with no prior programming background

File: stationarity_adf_ar.py
Repository: https://github.com/skimeur/QMF

License: MIT (code)
Year: 2026
Author: Eric Vansteenberghe
"""


import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats
from scipy.optimize import minimize
from statsmodels.tsa.arima.model  import ARIMA
from statsmodels.tsa.stattools import adfuller

# to plot, set ploton to ploton to 1
ploton = False

# change the working directory
os.chdir('//Users/skimeur/Mon Drive/QMF/')

#%% Presenting the concept of unit root

# we define a time series following a process with a unit root
thetai = 1
# start a list with y(0) = 10**3
yt = [10**2]
# we define a shock happening at mid period
shocky = 50
# length of our data
lent = 1000
for i in range(1,lent):
    if i == int(lent/2):
        yt.append(( thetai * yt[i-1] +  shocky + np.random.normal(0,1,1))[0])
    else: 
        yt.append((thetai *   yt[i-1] + np.random.normal(0,1,1))[0])

dfy = pd.DataFrame(yt) 
dfy.columns = ['yt']

# now compute and plot the rate of change of y and show that it was not impacted by the change except on the date of the impact
dfy['r_yt'] = (dfy.yt - dfy.yt.shift(1)) / dfy.yt.shift(1)

if ploton:
    ax = dfy.yt.plot(title='time series with a unit root and a shock at mid period')
    fig=ax.get_figure()
    fig.savefig('fig/unitrootjump.pdf')
    plt.close()
    ax = dfy['r_yt'].plot(title='returns of the time series')
    fig=ax.get_figure()
    fig.savefig('fig/rateunitrootjump.pdf')
    
# Augmented Dickey-Fuller test: is there a unit root?
# H0: there is a unit root
adfuller(dfy.yt, regression='n')
adfuller(dfy.yt, regression='c')
# imposing 'n' to regression mean that we assume a random walk
# imposing 'c' means you assume a random walk with a drift
# p-value high in any case, we have a high probability to wrongly reject H0 we do notreject H0
# we assume thate there is a unit root
adfuller(dfy['r_yt'].dropna(), regression='n')
adfuller(dfy['r_yt'].dropna(), regression='c')

del dfy, i, lent, shocky, yt, thetai

#%% AD Fuller test on world population
# https://ourworldindata.org/grapher/population?country=~OWID_WRL&overlay=download-data
# Inspect sheets (MPD files sometimes store data in a specific sheet)
pop = pd.read_csv('data/population_HYDE.csv')
pop = pop.loc[:,['Year', 'Population (historical)']]
pop = pop.loc[pop.Year>=1500,:]
pop = pop.set_index("Year").sort_index()
year_grid = np.arange(pop.index.min(), pop.index.max() + 1, 10)  # 10-year grid
pop10 = pop.reindex(year_grid).interpolate("linear")
y = np.log(pop10["Population (historical)"]).dropna()
y.plot()

adfuller(y, regression="ctt", autolag="AIC", maxlag=10)
# after removing a quadratic trend, the residual behaves like a stationary process


#%% Exercise with the AR(1)


#%% Granger and Newbold 1974 idea
series1 = np.random.normal(0,1,10**3)
series2 = np.random.normal(0,1,10**3)

df = pd.DataFrame(np.matrix([series1,series2]).transpose())
df.columns = ['s1','s2']
    
# regress one series on the other
results_prelim = smf.ols('s1 ~ s2',data=df).fit()
results_prelim.summary()  

#%% Illustration of a spurious regression  

yt = [0]
xt = [0]
# length of our data
lent = 10**3
for i in range(1,lent):
    yt.append((yt[i-1] + np.random.default_rng(i).normal(0,1,1))[0])
    xt.append((xt[i-1] + np.random.default_rng(i+10**3+lent).normal(0,1,1))[0])

dfspur = pd.DataFrame(np.matrix([yt,xt]).transpose())
dfspur.columns = ['yt','xt']

# regress one series on the other
results_spur = smf.ols('yt ~ xt',data=dfspur).fit()
results_spur.summary()

# numerator of the OLS estimator
num_beta = (dfspur['xt'] * (dfspur['yt']-dfspur['yt'].mean())).sum()
# denominator of the OLS estimator
den_beta = (dfspur['xt']* (dfspur['xt'] - dfspur['xt'].mean()) ).sum()

# we find indeed the OLS estimated beta
# by "luck" we have a huge numerator and denominator that "compensate"
ratio = num_beta / den_beta

# estimator variance
sigma_hat = results_spur.resid.std()**2
estimator_Variance = sigma_hat / den_beta
# this variance is artificially low because of "huge" denominator

# this is the intuition behind why a spurious regression can provide artificially significant OLS relationship

#%% Before regressing, a spurious regression

# we create two independent variables
# we loop over several model generation and compute the mean R square

# number of loops
looplen = 20

# to store the adjusted Rsquares
Rsquarelist = []

for loopi in range(0,looplen):
    yt = [0]
    xt = [0]
    # length of our data
    lent = 10**3
    for i in range(1,lent):
        yt.append((yt[i-1] + np.random.normal(0,1,1))[0])
        xt.append((xt[i-1] + np.random.normal(0,1,1))[0])
    
    dfspur = pd.DataFrame(np.matrix([yt,xt]).transpose())
    dfspur.columns = ['yt','xt']
    
    # regress one series on the other
    results_spur = smf.ols('yt ~ xt',data=dfspur).fit()
    
    #results_spur.summary()
    Rsquarelist.append(results_spur.rsquared_adj)

# we expect the R^2 to be close to zero, but we find that this is not the case
np.mean(Rsquarelist)
np.max(Rsquarelist)

# Example with the last random draw:
results_spur.summary()
dfspur.corr()
if ploton:
    ax = dfspur.plot(title='Two independent time series, with both a unit root')
    fig = ax.get_figure()
    #fig.savefig('fig/spuriousillustration.pdf')

del looplen, Rsquarelist, yt, xt, lent, dfspur, loopi, i
        

#%% AR(2), non stationary process with a unit root

# we define an AR(2) process
# process length AR2length
AR2length = 100
ARp2 = []
alphaAR2 = 0
beta1 = 1.6
beta2 = -0.6
# check that beta1 + beta2 == 1
beta1 + beta2 == 1
# if you want to convince yourself of the non-stationarity, you can input a small shock and the effect doesn't vanish
epsilon0 = 10**(-10)
epsilon1 = 0
ARp2.append(alphaAR2 + epsilon0)
ARp2.append(alphaAR2 + beta1 * ARp2[0] + epsilon1)
for ti in range(2,AR2length):
    ARp2.append(alphaAR2 + beta1 * ARp2[ti-1] +  beta2 * ARp2[ti-2])

# plot our AR(2) process
if ploton:
    plt.plot(ARp2)

# compute the matrix of the AR(2) process in a VAR form
ARp2matrix = pd.DataFrame([[beta1, beta2],[1,0]])

# compute the eigenvalues of the process
np.abs(np.linalg.eig(ARp2matrix)[0]) # there is a unit root
if ploton ==1:
    plt.plot(pd.DataFrame(ARp2).diff())
    
# Augmented Dickey-Fuller test: is there a unit root?
# H0: there is a unit root
adfuller(ARp2[20:], regression='n')
# imposing 'n' to regression mean that we assume a random walk
adfuller(ARp2[20:], regression='c')
# imposing 'c' means you assume a random walk with a drift
# NB: here we remove the first 20 observations as there are just noize before the signal stabilize
# this is close to the concept of burn in

# as an illustration, if we do not remove the initial 20 first observations, our conclusion differs:
adfuller(ARp2, regression='n')
adfuller(ARp2, regression='c')

del ARp2, ARp2matrix, alphaAR2, beta1, beta2, epsilon0, epsilon1, ti

# exercise: change the values of beta1 and beta2 to have a stationary process
