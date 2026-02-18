#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Structural VAR (SVAR) Analysis of GDP Growth and Inflation
===========================================================

Author: Eric Vansteenberghe
Lecture: Quantitative Methods in Finance (QMF)
License: MIT License

Description
-----------
This script estimates a Structural Vector Autoregression (SVAR) model
to analyze the dynamic interactions between GDP growth and inflation.

The workflow includes:

1. Data Import and Cleaning
   - Inflation data imported from Banque de France Webstat export.
   - French month names manually mapped to standard datetime format.
   - Monthly inflation aggregated to annual frequency (mean).
   - GDP data imported and transformed into annual growth rates.
   - Final dataset constructed with aligned GDP growth and inflation.

2. Structural Identification
   - A two-variable SVAR model is estimated using the A-matrix form.
   - Identification restriction: no contemporaneous impact of inflation
     on GDP (short-run restriction).
   - Model estimated via maximum likelihood (Nelder–Mead solver).

3. Diagnostics
   - Stability check of the VAR system.
   - Residual normality test.
   - Residual autocorrelation (whiteness) test.
   - Autocorrelation plots of residuals.

4. Impulse Response Functions (IRFs)
   - Structural impulse responses plotted when `ploton = True`.
"""

import pandas as pd
import os
import numpy as np
from statsmodels.tsa.api import SVAR

# to plot, set ploton to ploton to 1
ploton = True

# change the working directory
os.chdir('//Users/skimeur/Mon Drive/QMF/')

#%% import data
# import inflation
df = pd.read_csv('data/Webstat_Export_20210113-2.csv', sep=';', encoding='latin1', skiprows=[0, 1, 2, 3, 4], index_col=0)
df= df.iloc[::-1]
df = df.replace({',': '.'}, regex=True) 
df.columns = ['value']
df = pd.DataFrame(pd.to_numeric(df.value, errors='coerce').dropna())

# Reset the index so we can work with 'Source :' column
df = df.reset_index()

# Rename the 'Source :' column to a more usable name
df = df.rename(columns={'Source :': 'date_str'})

# Convert 'date_str' to datetime by manually mapping French month names
month_map = {
    'Jan': 'Jan', 'Fév': 'Feb', 'Mar': 'Mar', 'Avr': 'Apr', 'Mai': 'May', 'Jun': 'Jun',
    'Jul': 'Jul', 'Aoû': 'Aug', 'Sep': 'Sep', 'Oct': 'Oct', 'Nov': 'Nov', 'Déc': 'Dec'
}

# Standardize month names
for french, english in month_map.items():
    df['date_str'] = df['date_str'].str.replace(french, english)

# Convert to datetime
df['date'] = pd.to_datetime(df['date_str'], format='%b %Y', errors='coerce')

# Drop any rows with invalid dates if necessary
df = df.dropna(subset=['date'])

# Set 'date' as the index
df = df.set_index('date')

# Keep only the 'value' column
df = df[['value']]

df['year'] = df.index.year
df = df.groupby('year')['value'].mean()
df.index = df.index.astype(int)

# import GDP
gdp = pd.read_csv('data/Webstat_Export_20210113.csv', sep = ';', encoding = 'latin1', skiprows = [0,1,2,3,4], index_col=0)
gdp = gdp.iloc[::-1]
gdp = gdp.replace({',': '.'}, regex=True) 
gdp = gdp.astype(float)
gdp.index = gdp.index.astype(int)
gdp = gdp/gdp.shift(1) - 1

df = pd.concat([gdp, df], axis=1)
df = df.dropna()
df.columns = ['GDP', 'inflation']
del gdp


#%% SVAR

# we impose no short-run impact of inflatin on GDP
A = np.array([[1, 0], ['E', 1]])

df.columns = ['GDP', 'inflation']
svarmodel = SVAR(df, svar_type='A', A=A)
res_svar = svarmodel.fit(maxlags=1, trend='c',solver='nm')

# SVAR stabile?
res_svar.is_stable()

# Normality test of residuals
print(res_svar.test_normality())

# plot time series autocorrelation functions
res_svar.plot_acorr()
 
# residuals autocorrelation test
print(res_svar.test_whiteness())

np.linalg.inv(res_svar.A)
    
# IRF
if ploton:
    res_svar.irf().plot()
