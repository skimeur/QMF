#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Eric Vansteenberghe
Quantitative Methods in Finance
Beginner exercise with pandas DataFrames - part 1
Clean, rename columns, convert date information to python recognizable date format
Compute growth rates
Plot your data
Descriptive statistic of your data
2024
"""

import pandas as pd

ploton = False


#%% Create manually a data frame

# we create a data frame with the evolution of the French population
listtest = [66790,66763,66735,66710,66688,66672,66659,66644,66628]
pop = pd.DataFrame(listtest)

# always useful to plot the data for visual inspection
#pop.plot()
pop.plot()

# I don't think that the French population is declining:
# in fact we inversed the data, upside down
pop = pop.iloc[::-1]


#%% How to reverse a list (then an index) and how it works

# create a simple list
mylist = range(0,4)
# we wanted a list object
mylist = list(mylist)
# or more directly
mylist = list(range(0,4))
mylist[0:1:1]
mylist[0:2:1]
mylist[0:3:1]
mylist[0:4:1]
mylist[0:len(mylist):1]
# step of 2
mylist[0:len(mylist):2]
# omit the start and end, so use the default in python's
mylist[::]
# reverse the step (-1)
mylist[::-1]
# if we want to define start:end:step the stop is not obvious
mylist[3::-1]

#%% Plot our data set
# we can try to plot
#pop.plot()
# what you see is that now the index is incorrect
pop.reset_index(drop = True, inplace = True)
# equivalently
# pop = pop.reset_index(drop = True)
if ploton:
    pop.plot()

# now we want to use the calendar dates as index
dates = pd.date_range('2016-01', '2016-09', freq='MS')

pop.index = dates

pop.columns = ['Population']

if ploton:
    pop.plot(title='French population in thousands')

# pop['Population_t_minus_1'] = pop.Population.shift(1)
# pop['gr'] = pop['Population'] / pop['Population_t_minus_1'] - 1
# pop.dropna(inplace=True, how='any')
# compute the monthly population change in France and plot it
change_pop = 100 * (pop - pop.shift(1))/ pop.shift(1)
if ploton:
    change_pop.plot(title = 'Monthly French population change in percent')

