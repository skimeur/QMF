#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vansteenberghe_robust_regressions.py
QMF 2026 — Robust Regression Methods (French GDP & Population)

This script accompanies the QMF lecture notes section on robust regression.
It compares several estimation techniques when outliers are present in a 
bivariate macroeconomic relationship (French GDP growth and Population growth).

The workflow includes:

1) Data preparation
   - French population (INSEE export)
   - French GDP
   - Construction of stationary growth rates

2) Baseline estimation
   - Ordinary Least Squares (OLS)

3) Outlier treatment
   - Dummy-variable correction
   - OLS with influential observation control

4) Robust alternatives
   - Median regression (Quantile regression, q=0.5)
   - Robust Linear Model (Huber M-estimator)
   - RANSAC (Random Sample Consensus, sklearn)

5) Visualization
   - Comparison of fitted lines across estimators
   - Identification of inliers and outliers (RANSAC)

6) Output formatting
   - LaTeX regression tables via Stargazer

Pedagogical objectives
----------------------
- Illustrate sensitivity of OLS to extreme observations.
- Compare L2 (OLS), L1 (median), M-estimation (Huber), and RANSAC approaches.
- Highlight robustness-efficiency trade-offs in finite samples.

Author: Eric Vansteenberghe
Course: Quantitative Methods in Finance
Year: 2026
License: MIT (recommended for GitHub publication)
"""


import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.formula.api as smf
from stargazer.stargazer import Stargazer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor

# to plot, set ploton to ploton to 1
ploton = True

# change the working directory
os.chdir('//Users/skimeur/Mon Drive/QMF/')

#%% Import the data on French population again as in part 1
# Load the CSV file 'Valeurs.csv' with a semicolon separator, Latin-1 encoding, skip the first three rows, and set no header or index column
pop = pd.read_csv('data/Valeurs.csv', sep=';', encoding='latin1', skiprows=[0,1,2], header=None, index_col=False)

# Reverse the DataFrame rows to make the data appear in chronological order, it was initially reversed
pop = pop.iloc[::-1]

# Rename columns to 'Year', 'Month', and 'Population' for clarity
pop.columns = ['Year', 'Month', 'Population']

# Set the index to a monthly date range starting from January 1994 to October 2016
pop.index = pd.date_range('1994-01', '2016-10', freq='M')

# Drop the 'Year' and 'Month' columns as they are no longer needed after setting the date index
pop = pop.drop(columns=['Year', 'Month'])

# Replace any spaces in the 'Population' column with an empty string to clean the data
pop = pop.replace({' ': ''}, regex=True)

# Convert the 'Population' column values to floats for numerical operations
pop = pop.astype(float)

# Scale the 'Population' values down by dividing by 1000, so it is in millions
pop = pop / 1000

decomposition = seasonal_decompose(pop["Population"], 
                                    model="additive", 
                                    period=12)

# Deseasonalized population
pop["Population"] = pop["Population"] - decomposition.seasonal

#%% Wage sum, CVS = “Corrigée des Variations Saisonnières” → Seasonally adjusted.
# here, we call gdp in fact what is "Wage Sum in Branches"
gdp = pd.read_csv('data/GDP.csv',sep=';',encoding='latin1',skiprows = [0,1])
gdp = gdp.iloc[::-1]
gdp.columns = ['Quarter','GDP']
gdp.index = pd.date_range('1949-01', '2016-09', freq='Q') 
gdp = gdp.drop(columns=['Quarter'])
gdp = gdp.replace({' ': ''}, regex=True) 
gdp = gdp.astype(float)

#%% concatenate both time series into one data frame
df = pd.concat([pop.Population,gdp],axis=1)

# drop rows with missing values
df = df.dropna(axis=0)

# compute quarterly changes
dx = (df-df.shift(1))/df.shift(1)
dx = dx.dropna()

del gdp

#%% OLS
# we can do a linear regression on stationary series, the returns
olsreg = smf.ols('Population ~ GDP', data = dx).fit()
olsreg.summary()

#%% OLS without the two outliers
# outlier with the maximum growth rate of population
outlier = pd.DataFrame(dx.sort_values(by='Population', ascending=False).iloc[0,:])

# create dummy variable for both outlier
dx['dummy'] = 0
dx.loc[outlier.columns,'dummy'] = 1

olsreg_wo_outliers = smf.ols('Population ~ GDP + dummy', data = dx).fit()
olsreg_wo_outliers.summary()

#%% Median regression
quantreg = smf.quantreg('Population ~ GDP',data = dx).fit(q=.5)
quantreg.summary()
quantreg_wo_outliers = smf.quantreg('Population ~ GDP + dummy',data = dx).fit(q=.5)
quantreg_wo_outliers.summary()

#%% Robust linear model

dx['intercept'] = 1

rlm_model = sm.RLM(dx.Population.values, dx.loc[:,['intercept','GDP']].values, M=sm.robust.norms.HuberT()).fit()
rlm_model.summary()

rlm_model_wo_outliers = sm.RLM(dx.Population.values, dx.loc[:,['intercept','GDP','dummy']].values, M=sm.robust.norms.HuberT()).fit()
rlm_model_wo_outliers.summary()

#%% RANSAC

ransac = RANSACRegressor(LinearRegression(), random_state=0,residual_threshold=0.002)
ransac.fit(dx.loc[:,['GDP']].values, dx.Population.values)
ransac.estimator_
print('RANSAC Slope: %.3f' % ransac.estimator_.coef_[0])
print('RANSAC Intercept: %.3f' % ransac.estimator_.intercept_)

inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

#%% Visual of the regression

# plot the data
stepsize = 0.001
x = np.arange(1.1*dx['GDP'].min(),1.1*dx['GDP'].max(),stepsize)
if ploton:
    ransacy = ransac.predict(x[:, np.newaxis])
    axes = plt.gca()
    axes = plt.scatter(dx['GDP'][inlier_mask], dx['Population'][inlier_mask],c='steelblue', edgecolor='white', marker='o', label='Inliers')
    axes = plt.scatter(dx['GDP'][outlier_mask], dx['Population'][outlier_mask],c='limegreen', edgecolor='white', marker='s', label='Outliers')
    #axes = plt.scatter(dx['GDP'],dx['Population'])
    axes = plt.xlabel('French GDP changes')
    axes = plt.ylabel('French Population changes')
    axes = plt.plot(x, olsreg.params[0] + olsreg.params[1] * x ,'-', color='k', label='OLS')
    axes = plt.plot(x, olsreg_wo_outliers.params[0] + olsreg_wo_outliers.params[1] * x ,'-', color='g', label='OLS no outliers')
    axes = plt.plot(x, quantreg.params[0] + quantreg.params[1] * x ,'-', color='r', label='median reg')
    axes = plt.plot(x, rlm_model.params[0] + rlm_model.params[1] * x ,'-', color='b', label='linear robust reg')
    axes = plt.plot(x, ransacy,  '--', color='m', label='RANSAC') 
    axes = plt.ylim(0,0.0045)
    axes = plt.xlim(-0.013,0.02)
    axes = plt.legend(loc = 'upper left')
    figaxes = axes.get_figure()
    figaxes.savefig('fig/GDP_Pop_robustreg.pdf')
    del axes

del x, stepsize

#%% output of regressions
stargazer = Stargazer([olsreg, olsreg_wo_outliers])
regout = stargazer.render_latex()

stargazer2 = Stargazer([olsreg, olsreg_wo_outliers,quantreg, quantreg_wo_outliers])
regout2 = stargazer2.render_latex()
