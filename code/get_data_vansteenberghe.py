#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Eric Vansteenberghe
Quantitative Methods in Finance

Getting empirical data

2021
"""

from datetime import datetime
import pandas as pd
import os
# if not installed, open a cmd, and:
# pip install pandas_datareader
#import quandl
#import fix_yahoo_finance as yf
#from pandas_datareader import data as pdr
#yf.pdr_override()

# if internet connection is running and Yahoo up
internet = 0

#We set the working directory (useful to chose the folder where to export output files)
os.chdir('/Users/skimeur/Mon Drive/QMF')

df = pd.read_csv('data/Webstat_Export_20210113.csv')

# ploting data can take time, so I prefer using a variable to decide whether I plot when I execute or not
ploton = False

#%% Banque de France
""" Property prices in France from Banque de France webstat"""

# with pandas, you can read csv with the read_csv() command
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
immo = pd.read_csv('data/france_property_prices_BdF.csv', sep = ';', encoding = 'latin1', skiprows = [0,1,2,3,4], decimal=',')

immo.columns = ['date','prices']

immo = immo.iloc[::-1]

immo.index = pd.date_range('1996-03', '2017-03', freq='QE') 

del immo['date']
#immo = immo.replace({',': '.'}, regex=True) 
#immo.dtypes
#immo = immo.astype(float)

if ploton:
    ax = immo.plot(title='French property prices', legend=False)
    fig = ax.get_figure()
    fig.savefig('fig/frenchpropprices.pdf')

del immo

# UPDATE exercise
immo = pd.read_csv('data/Webstat_Export_20200904.csv', sep = ';', encoding = 'latin1', skiprows = [0,1,2,3,4])

immo.columns = ['date','prices']

immo = immo.iloc[::-1]

# automatic date reading
# first we need to replace the T by a Q
immo['date'].replace({'T':'Q'}, inplace=True, regex=True)
# then we want to write the dates in the correct format YYYY-Q1
# one lengthy way to do it, inspired by what we did in class:
immo['quarter'] = immo['date'].str[1]
immo['year'] = immo['date'].str[3:]
immo['Q'] = '-Q'
immo['date'] = immo['year'] + immo['Q']  + immo['quarter']
immo['date'] = pd.to_datetime(immo['date'])
# or there is this one liner:
#immo['date'] = pd.to_datetime(immo['date'].str.replace(r'(Q\d) (\d+)', r'\2-\1'), errors='coerce')

immo.index = immo['date']


immo = immo.replace({',': '.'}, regex=True) 
immo.dtypes
immo['prices'] = immo['prices'] .astype(float)

if ploton:
   immo.prices.plot(title='French property prices')

del immo

# exercise
# some lock-down dates and how to illustrate them
# frlockeddate = pd.to_datetime('2020-03-17')
# frlockend = pd.to_datetime('2020-05-11')
# dflockdown['france'] = 0
# dflockdown.loc[(dflockdown.index < frlockend)&(dflockdown.index > frlockeddate), 'france'] = 1
# dflockdown.loc[(dflockdown.index > frdeuxiem)&(dflockdown.index < frdeuxiemend), 'france'] = 1


# if ploton:
# plot the df as usuall
#     ax = df.plot()
# fill with specific colors
#     ax.fill_between(dflockdown.loc[dflockdown.index>=pd.to_datetime('2019-12-31'),:].index, ymin, ymax, where=dflockdown.loc[dflockdown.index>=pd.to_datetime('2019-12-31'),'france'].values, color='blue', alpha=0.4)
#     fig = ax.get_figure()
#     fig.savefig("fig/yourfilename.pdf")
#     plt.close()


#%% Updated Banque de France French real estate prices

df = pd.read_csv('data/RPP_20220317.csv', sep = ';', encoding = 'latin1', skiprows = [0,1,2,3,4], index_col=0).iloc[::-1]
df.index = pd.to_datetime(df.index.str.replace(r'(Q\d) (\d+)', r'\2-\1'), errors='coerce')
df = df.replace({',': '.'}, regex=True) 
df = df.astype(float)

if ploton:
    df.plot()

# base 100 in 2014
df = df / df.loc[pd.to_datetime('2014-01-01'),:]

#%% INSEE data

# import INSEE french population data
frenchpop = pd.read_csv('data/frenchpop.csv',sep = ';',encoding = 'latin1',skiprows = [0,1])
frenchpop = frenchpop.iloc[:,:2]
frenchpop.columns = ['date','pop']

frenchpop = frenchpop.iloc[::-1]

frenchpop.index = pd.date_range('1975-01', '2017-07', freq='MS') 

frenchpop = frenchpop['pop']

if ploton:
    ax = frenchpop.plot(title='French population', legend=False)
    fig = ax.get_figure()
    fig.savefig('fig/frenchpopINSEE.pdf')

del frenchpop

#%% OECD data: industrial production
oecd_data = pd.read_csv('data/MEI_ARCHIVE_01112020095601033.csv', encoding='latin1')

oecd_dataFR = oecd_data.loc[oecd_data.Country=='France', :]

uniquevarnames = list(oecd_data.Variable.unique())
col3 = oecd_data.loc[:, ['Country','TIME','Value']]
# pivot table
oecd = pd.pivot_table(oecd_data, values='Value', index='TIME', columns=['Country'], aggfunc='mean')
# index to date format
oecd.index = pd.to_datetime(oecd.index, format='%Y-%m')

# show where the missing values are
missingcount = len(oecd) - oecd.count()
missingcount = list(missingcount[missingcount<=1].index)
missingcount

# sort by longest time series
oecd = oecd.loc[:,missingcount]
oecd.dropna(how='any', inplace=True)

if ploton:
    ax = oecd.loc[:,['Germany', 'United Kingdom', 'United States']].plot()
    fig = ax.get_figure()
    fig.savefig('fig/oecd_ip.pdf')

#%% FRED
""" USA macroeconomic time series from FRED """
# Inspired by Introduction to Python for Econometrics, Statistics and Data Analysis by Kevin Sheppard

# I list the code found on FRED: https://fred.stlouisfed.org/
codes = ['GDPC1','INDPRO','CPILFESL','UNRATE','GS10','GS1','BAA','AAA','PCEC96','BOGMBASEW','T10Y2Y']
# I list what the code corresponds to
names = ['Real GDP','Industrial Production','Core CPI','Unemployment Rate','10 Year Yield','1 Year Yield','Baa Yield','Aaa Yield','real personal consumption expenditure','Monetary Base','difference between two- and 10-year Treasury yields']
# r to disable escape
base_url = r'https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}'
# define the starting date of observations
starti = datetime(1919, 1, 1)
# define an ending date
endi = datetime(2020, 9,4)

# create a data frame to store the data
fred = pd.DataFrame(index=pd.date_range(start=starti, end=endi, freq='MS'))
# do a loop over all codes
for code in codes:
    print(code)
    # define the url for download
    url = base_url.format(code=code)
    # import the data in a intermediary data frame
    dfinter = pd.read_csv(url)
    # convert the date to python date format
    dfinter.observation_date = pd.to_datetime(dfinter.observation_date)
    # place the date information in the index
    dfinter.index = dfinter.observation_date
    # drop the DATE column
    dfinter = dfinter.drop('observation_date',axis=1)
    # join the intermediary data frame with the fred data frame
    fred = fred.join(dfinter)

del url, base_url, starti, endi, dfinter, code, codes

# rename columns
fred.columns = names
#fred = fred.apply(pd.to_numeric,errors='corece')

if ploton:
    fred['Monetary Base'] = fred['Monetary Base'] / 1000
    ax = fred['Monetary Base'].dropna().plot(title='USA: monetary base, Bns USD')
    fig = ax.get_figure()
    fig.savefig('monetarybase.pdf')

# deal with empty data
#imp = Imputer(missing_values='NaN', strategy='mean', axis=1)
#imputed_column = imp.fit_transform(fred['Real GDP'].values.reshape(1,-1)).T
#fred.loc[:,['Real GDP']] = imputed_column
if ploton:
    fred.loc[:,['Real GDP']].dropna().plot()
    ax = fred.iloc[:,1:3].plot()
    fig = ax.get_figure()
    fig.savefig('fig/indprodCPI.pdf')
    fred.loc[:,'Unemployment Rate'].plot()


#%% Inflation, GDP GR, FED Fund Rate

# I list the code found on FRED: https://fred.stlouisfed.org/
codes = ['GDPC1','CPILFESL','FEDFUNDS']
# r to disable escape
base_url = r'https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}'
# define the starting date of observations
starti = datetime(2020, 5, 1)
# define an ending date
endi = datetime(2024, 9,30)

# create a data frame to store the data
fred = pd.DataFrame(index=pd.date_range(start=starti, end=endi, freq='MS'))
# do a loop over all codes
for code in codes:
    print(code)
    # define the url for download
    url = base_url.format(code=code)
    # import the data in a intermediary data frame
    dfinter = pd.read_csv(url)
    # convert the date to python date format
    dfinter.observation_date = pd.to_datetime(dfinter.observation_date)
    # place the date information in the index
    dfinter.index = dfinter.observation_date
    # drop the DATE column
    dfinter = dfinter.drop('observation_date',axis=1)
    # join the intermediary data frame with the fred data frame
    fred = fred.join(dfinter)

del url, base_url, starti, endi, dfinter, code, codes

# Resample quarterly
fred = fred.resample('QS').mean()

for coli in ['GDPC1','CPILFESL']:
    fred[coli] = (fred[coli] - fred[coli].shift(1)) / fred[coli].shift(1)

# rename columns
# I list what the code corresponds to
names = ['Real GDP gr','Core Inflation','Fed Fund Rate']

fred.columns = names
fred.dropna(inplace=True, how='all', axis=1)
fred.dropna(inplace=True, how='all', axis=0)

if ploton:
    fred.plot(secondary_y='Fed Fund Rate')

#%% EXERCISE # reproduce the graphic from the Financial Times Article "Powell downplays recession fears that yield curve is said to signal" of Thursday 19th July 2018
# I list the code found on FRED: https://fred.stlouisfed.org/
codes = ['GDPC1','INDPRO','CPILFESL','UNRATE','GS10','GS1','BAA','AAA','PCEC96','BOGMBASEW','T10Y2Y','GS2']
# I list what the code corresponds to
names = ['Real GDP','Industrial Production','Core CPI','Unemployment Rate','10 Year Yield','1 Year Yield','Baa Yield','Aaa Yield','real personal consumption expenditure','Monetary Base','difference between two- and 10-year Treasury yields','2-Year Treasury Constant Maturity Rate']
# r to disable escape
base_url = r'https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}'
# define the starting date of observations
starti = datetime(2000, 1, 1)
# define an ending date
endi = datetime(2020, 10,1)

# create a data frame to store the data
fred = pd.DataFrame(index=pd.date_range(start=starti, end=endi, freq='MS'))
# do a loop over all codes
for code in codes:
    print(code)
    # define the url for download
    url = base_url.format(code=code)
    # import the data in a intermediary data frame
    dfinter = pd.read_csv(url)
    # convert the date to python date format
    dfinter.observation_date = pd.to_datetime(dfinter.observation_date)
    # place the date information in the index
    dfinter.index = dfinter.observation_date
    # drop the DATE column
    dfinter = dfinter.drop('observation_date',axis=1)
    # join the intermediary data frame with the fred data frame
    fred = fred.join(dfinter)

del url, base_url, starti, endi, dfinter, code, codes

# rename columns
fred.columns = names
#fred = fred.apply(pd.to_numeric,errors='corece')
fred.columns
fred['FTplot'] = fred['10 Year Yield'] - fred['2-Year Treasury Constant Maturity Rate']
if ploton:
    fred['FTplot'].dropna().plot()
# add label: Feb 2000: the curve inverted before the tech crash.
# dec 2005: an early warning ahead of the 2008 financial crisis
# present day: the yield curve is flattering again

# Are both data series similar?
# first convert the serie to numeric (force strange data points to np.NaN using errors='corece')
fred['difference between two- and 10-year Treasury yields'] = pd.to_numeric(fred['difference between two- and 10-year Treasury yields'], errors='coerce')

if ploton:
    fred.loc[:,['FTplot','difference between two- and 10-year Treasury yields']].dropna().plot()

#%% CAC 40 data from Euronext
# one can download the index history from the producer, Euronext
# https://www.euronext.com/en/indices/index-statistics?archive=this_year

cac = pd.read_csv('data/CAC 40_quote_chart.csv', index_col=0)
cac.index = pd.to_datetime(cac.index)
del cac['volume']
cac.columns = ['CAC 40 close']

cacnet = pd.read_csv('data/CAC 40 NR_quote_chart.csv', index_col=0)
cacnet.index = pd.to_datetime(cacnet.index)
del cacnet['volume']
cacnet.columns = ['CAC 40 total return close']
cacnet = pd.concat([cac, cacnet], axis=1)
cacnet = cacnet.dropna(how='any')
cacnet = 100 * cacnet / cacnet.iloc[0,:]

if ploton:
    ax = cac.plot()
    fig = ax.get_figure()
    fig.savefig('CAC40close.pdf')
    ax2 = cacnet.plot()
    fig2 = ax2.get_figure()
    fig2.savefig('CAC40_net_close.pdf')
    
# compute daily return of the cac and cac net return
dx = (cacnet - cacnet.shift(1)) / cacnet.shift(1)
# after the Quantitative Easing announcement in Jan 2015
dxQE = dx.loc[dx.index >'2015-01-01']
# compute the average yearly growth rate
(1 + dx.mean())**252 - 1
# COVID-19 excluded
(1 + dx.loc[dx.index <'2020-01-01'].mean())**252 - 1

# compute the average yearly growth rate
(1 + dxQE.mean())**252 - 1


del cac, cacnet, dx, dxQE


#%% IMF data

imf = pd.read_excel('data/GDP_and_Components.xls', sheet_name = 'GDP and Components',skiprows = [0,1,2,3,4,5], index_col=1)

# work on that data to be able to compare any 2 countries GDPs and GDP growths, find systemic crises

"""
#%% Quandl

# First you need to install the Quandl module and get an API key from their website

# I store my API key in a seperated file
#from quandl_API import your_API

quandl.ApiConfig.api_key =  your_API

data = quandl.get("EOD/VZ",returns="pandas")
data2 = quandl.get("WIKI/AAPL", returns ="pandas")


tickers_list = pd.read_csv('R_Data/WIKI-datasets-codes.csv')
tickers = list(tickers_list.iloc[:,0])



#We define a start and end date, this will constitute the period over which we want to study the performance of certain shares:
starti = datetime(1970, 1, 1)
endi = datetime.now()



if internet == 1:
    quandl_data=pd.DataFrame(index=pd.date_range(start=starti,end=endi,freq='D'))
    for ticker in tickers:
        if not any(ticker in s for s in list(quandl_data.columns)):
            interm = quandl.get(ticker, returns ="pandas")
            interm.columns = [ticker + ' ' + 'Open', ticker + ' ' +'High', ticker + ' ' +'Low', ticker + ' ' +'Close', ticker + ' ' +'Volume',ticker + ' ' + 'Ex-Dividend', ticker + ' ' +'Split Ratio', ticker + ' ' +'Adj. Open', ticker + ' ' +'Adj. High', ticker + ' ' +'Adj. Low', ticker + ' ' +'Adj. Close', ticker + ' ' +'Adj. Volume']
            quandl_data = quandl_data.join(interm)
            quandl_data.to_csv('data/quandl_data.csv')

    quandl_data = quandl_data.dropna(axis=0, how = 'all')
    quandl_data.to_csv('data/quandl_data.csv')
else:
    quandl_data = pd.read_csv('quandl_data.csv')
#test = not any(ticker in s for s in list(quandl_data.columns))

#posti = tickers.index(ticker)

# keep only columns with a certain string: 'Adj. Close'
adjclose_cols = [col for col in quandl_data.columns if 'Adj. Close' in col]

quandl_adj = quandl_data.loc[:,adjclose_cols]


if ploton == 1:
    ax = quandl_adj.plot(title="Some quandl data")
    fig = ax.get_figure()
    fig.savefig('quandlextract.png')

del data, data2, interm, starti, endi, adjclose_cols, quandl_data, quandl_adj, your_API

#%% Yahoo finance


if internet == 1:
    #We define a start and end date
    starti = datetime(2006, 8, 1)
    endi = datetime.now()
    symboli = "DBK.DE"
    # download dataframe
    stocki = pdr.get_data_yahoo(symboli, start=starti, end=endi)
    #We create a vector, prixVW, with the closing price (so each price at the end of the trading day, around 17:30 in Europe):
    prixstocki = pd.DataFrame(stocki['Adj Close'])
    #We compute the daily return:
    rendementstocki = pd.DataFrame(prixstocki/prixstocki.shift(1) -1)
    
    prixstocki.plot()
    rendementstocki.plot()
    ax = stocki.loc[:,['High','Low']].plot(title='Deutsche Bank prices')
    fig = ax.get_figure()
    fig.savefig('DBhprices.PNG')
    
    del starti, endi, stocki, symboli, prixstocki, rendementstocki

    # fetch a list of indexes
    # CAC 40, S&P 500, Dow Jones industrial average, Nikkei 225, Dax, FTSE 100
    indexlist = ['^FCHI','^GSPC','^DJI','^N225','^GDAXI','^FTSE']
    indexnames = ['CAC40','S&P5000','DowJonesIA','Nikkei225','Dax','FTSE100']
    #We define a start and end date
    starti = datetime(1980, 1, 1)
    endi = datetime.now() 
    dates = pd.date_range(starti,endi,freq='D')
    dfindex = pd.DataFrame(index=dates)
    for symboli in indexlist:
        stocki = pdr.get_data_yahoo(symboli, start=starti, end=endi)
        stocki = pd.DataFrame(stocki['Adj Close'])
        stocki.columns = [symboli]
        dfindex = dfindex.join(stocki)
    dfindex.to_csv('dfindex.csv')
   
if plotonÒ:     
    dfindex.plot()

"""