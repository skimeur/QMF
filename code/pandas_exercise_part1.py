#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Eric Vansteenberghe
Quantitative Methods in Finance
Beginner exercise with pandas DataFrames - part 1
Select item by label or position
2021
"""

import pandas as pd
import numpy as np


# to plot, set ploton to ploton to True
ploton = False

#%% Create a simple DataFrame
df = pd.DataFrame([[1,1,2],[3,4,5],[7,8,9]])

# rename the columns of the dataframe
df.columns = ['A','B','C']

# average of columns the dataframe
df.mean(axis = 0)
# sum of lines of the dataframe
df.sum(axis = 1)


#%% Finding location of an element in a DataFrame

slice1 = df.loc[ : ,['A', 'C']]
slice2 = df.loc[1,['A', 'C']]
slice3 = df.loc[1: , :'B']

df.loc[1, 'A']

slice1i = df.iloc[ :,[0, 2]]
slice2i = df.iloc[1,[0, 2]]
slice3i = df.iloc[1:,:2]

#Row 2
df.iloc[2]
#Column C
df.T.iloc[2]

df.loc[1, 'A'] == df.iloc[1,0]

df_unstack = df.unstack()

df2 = df * 2

df.iloc[0] = 0

#%% Loop through a DataFrame

df3 = pd.DataFrame(index=[0,1,2], columns=[0,1,2])

for i in range(0,len(df)):
    for j in range(0,len(df.columns)):
        print("row",i,"col",j)
        df3.iloc[i,j] = df.iloc[i,j] * 2

df6 = pd.DataFrame(index=range(0,4),columns = range(0,4))

for i in range(0,len(df6)):
    for j in range(0,len(df6.columns)):
        df6.iloc[i,j] = i*j




