#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 12:57:46 2021

@author: Eric Vansteenberghe
Quantitative Methods in Finance
Beginner exercise with pandas DataFrames - part 1 - import data set
Import csv data sets
Clean, rename columns, convert date information to python recognizable date format
Compute growth rates
Matrix operations
Plot your data
Descriptive statistic of your data
"""

import pandas as pd
import os

# to plot, set ploton to ploton to True
ploton = False


#%% Import csv data as DataFrame

# if you don't know your pather to the data folder, try the following
# this function will start from a path and search for your file to return the name of the path to your file
def find_first(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)


# you will need to change '/Users/skimeur/' to your computer root
# this should be dsiplayed in the spyder top right "folder search" section
# either change skimer to your user name if you use Mac or C:// if you use windows
your_path = find_first('001641607.csv','/Users/skimeur')

# Once you found your path, you can copy and paste it below:

os.chdir('//Users/skimeur/Mon Drive/QMF/')

# https://www.insee.fr/fr/statistiques/serie/001641607?idbank=001641607
# Import French population data
df = pd.read_csv('data/001641607.csv', sep=';', encoding='latin1', skiprows=[0,1,2], header=None, index_col=False)
# flip my data frame (end becomes start)
df = df.iloc[::-1]
# rename the columns
df.columns = ['Year','Month','Population']
# chekc the type of each clumn
df.dtypes

df.index = pd.to_datetime((df.Year*100+df.Month).apply(str), format='%Y%m')
# an alternative would be to set the index manually, but you could have missing months so we do not recommend it:
#df.index    = pd.date_range('1994-01', '2016-10', freq='M') 
df = df.loc[:,['Population']]
# df = df.replace({' ': ''}) this would not work: https://stackoverflow.com/questions/45439506/pandas-replace-only-working-with-regex
df = df.replace({' ': ''}, regex=True) 
df = df.astype(float)
# INSEE indicates that the data is in 10^3, we might want to have the data in million
df = df / 10*3
if ploton:
    ax = df.plot(title='French population in million', legend=False)
    fig = ax.get_figure()
    fig.savefig('fig/frenchpopmonth.pdf')

dfyear = df.resample("YE").mean()

if ploton:
    dfyear.plot()

df_change = 100 * (df/df.shift(1)-1)

if ploton:
    ax = df_change.plot(title='Monthly change of French population in percent', legend=False)
    fig = ax.get_figure()
    fig.savefig('fig/frenchpopchange.pdf')

#%% Describe your data set
# describe the data set
statdesc = pd.concat([df.describe(), df_change.describe()], axis=1)
statdesc.columns = ['Population in million', 'Population growth rate in percentage']
# round up to 2 digits after the comma
statdesc = statdesc.round(2)
# Limit the index to our interest
statdesc = statdesc.loc[['count', 'mean', 'std', 'min', 'max'],:]
# export to latex to put in a report
statdesclatex= statdesc.to_latex(float_format="{:.2f}".format)
# Nota bene, in this very example, this is not very informative

#%% boxplot of the growthrate
df_change.boxplot(column=['Population'], sym='', showmeans=True)

#%% Input-Output exercise

# import the data set
M = pd.read_excel('data/Use_SUT_Framework_2007_2012_DET.xlsx', sheet_name="2012", skiprows=[0,1,2,3,4], engine='openpyxl', index_col=0)


