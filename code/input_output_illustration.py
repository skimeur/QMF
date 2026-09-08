#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified Input-Output example
Inspired from: http://ceadserv1.nku.edu/longa//classes/mat225/projects/Leontief.pdf

@author: Eric Vansteenberghe 
2022
"""

import pandas as pd
import numpy as np

# input-output matrix
Mgross = pd.DataFrame([[ 35,  5, 5] ,[5 , 60, 20],[10, 25, 40]])
Mgross.index = ['A','M','S']
Mgross.columns = ['A','M','S']

# D, the external (final) demand. The table is balanced: each row of Mgross plus D
# adds up to the gross output of the row's sector (35 + 5 + 5 + 40 = 85, etc.)
D = pd.DataFrame([40, 75, 145])
D.index = ['A','M','S']

# let's show the importance of naming index and columns
Dprime = pd.DataFrame([40,145,75])
Dprime.index = ['A','S','M']
# Dprime has the same information as D, right?

# we check this:
print(Mgross.dot(D))
print(Mgross.dot(Dprime))

# gross output
grossoutput = pd.DataFrame([85, 160, 220])
grossoutput.index = ['A','M','S']
# consumption matrix

# NOT the Leontief consumption matrix: a (3,1) array broadcasts along the rows, so each
# ROW i is divided by the gross output of sector i. This gives the allocation coefficients
# z_ij / p_i (share of i's sales going to j) of the supply-driven Ghosh model.
Mghosh = Mgross / grossoutput.values

print(Mghosh, 'row normalisation: Ghosh allocation coefficients, not what we want')

#%% Three ways of computing M correctly: divide each COLUMN j by the gross output p_j
# of the purchasing sector j, so that M_ij = z_ij / p_j is the input from i needed per
# unit of output of j (a Series aligns on the DataFrame's columns)
TgrossSeries = pd.Series([85, 160, 220])
TgrossSeries.index = ['A','M','S']


print(Mgross / TgrossSeries)

M = Mgross / TgrossSeries

# we duplicate the colonne to form a square matrix
#Tgross3x3 = grossoutput.T.append([grossoutput.T]*2,ignore_index=True)
Tgross3x3 = pd.concat([grossoutput, grossoutput, grossoutput], axis=1).T
Tgross3x3.index = ['A','M','S']
Tgross3x3.columns = ['A','M','S']

print(Mgross / Tgross3x3)


print(Mgross / grossoutput.T.values)

# vector product wouldn't do the job
Tg1 = 1 / TgrossSeries
print(Mgross.dot(Tg1))

I = pd.DataFrame(np.identity(len(M)))
I.index = ['A','M','S']
I.columns = ['A','M','S']
# compute P
print(np.linalg.inv(I- M).dot(D))
# or equivalently
print(np.linalg.inv(I- M) @ D)
P =  np.linalg.inv(I- M)@D

# checks: the economy is productive (column sums of M below one) and the model
# returns the gross output of the table when fed with its external demand
print('column sums of M:', M.sum(axis=0).round(2).to_dict())
print('P equals the gross output (85, 160, 220):', np.allclose(P, grossoutput.values))

#%% shock to agriculture
Dprime = D.copy(deep=True)
Dprime.loc['A',:] = Dprime.loc['A',:] + 1

Pprime = np.linalg.inv(I- M).dot(Dprime)

print(Pprime - P)

# NB: without any surprise, this also the first column of np.linalg.inv(I- M)

