#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Eric Vansteenberghe
Quantitative Methods in Finance
Beginner exercise with pandas DataFrames - part 1
Matrix operations
2021
"""

import pandas as pd
import numpy as np

# to plot, set ploton to ploton to True
ploton = False

#%% Matrix product 
# create a simple matrix
df = pd.DataFrame([[1,1,2],[3,4,5],[7,8,9]])

# rename the columns of the dataframe
df.columns = ['A','B','C']

# create a vector DataFrame, a columnc
colonne = pd.DataFrame([2,4,6])

# the index of colonne must be the df2 column name
colonne.index = df.columns

# perform a matrix times vector multiplication
out_mult = df.dot(colonne)

#%% Hadamard product

# we duplicate the colonne to form a square matrix
colonnereplicated = pd.concat([colonne,colonne,colonne], axis=1).T
# making sure the index and columns coincide
colonnereplicated.index = df.index
colonnereplicated.columns = df.columns
#deprecated: colonnereplicated = colonne.T.append([colonne.T]*2,ignore_index=True)

# multiply one df by the other, element by element
out_mult2 = df * colonnereplicated

#%% Inverse of the matrix
# we compute the inverse of the matrix
dfinv = np.linalg.inv(df)
# the product of a matrix (if non-singular) with its inverse is the identity matrix
print(df.dot(dfinv))

#%% Matrix product, eigenvalue, eigenvector and stability

# compute the eigenvalues and eigenvector of the DataFrame
eigen_df = np.linalg.eig(df)

eignevalues = eigen_df[0]
eigenvectors = eigen_df[1]
lambda2 = eignevalues[1]
V2 = eigenvectors[:,1]
# theory
df.dot(V2) 
lambda2 * V2
eigen_df[0][1] * eigen_df[1][:,1]
df.dot(eigen_df[1][:,1])

# extract the eigenvectors
EVlist = pd.DataFrame(eigen_df[1])
# extract the first eigenvector
firstEV = pd.DataFrame(EVlist.iloc[:,0])
firstEV.index = df.columns

# Matrix * eigenvector == eigenvalue * eigenvector
df.dot(firstEV)
firstEV * eigen_df[0][0]

# eienvalues 
eigenvalues_df = eigen_df[0]
# are some eigenvalues greater than 1? How many?
sum(eigenvalues_df > 1)

del colonne, dfinv, out_mult, out_mult2, colonnereplicated, eigenvectors, eignevalues, lambda2, V2, df, eigen_df, eigenvalues_df, EVlist, firstEV

#%% Matrix product and stability

# Network of exposures:
dfa = pd.DataFrame([[0,0.4,0.3],[0.5,0,0.4],[0.6,0.1,0]])

# create the identity matrix
ident_mat = pd.DataFrame([[1,0,0],[0,1,0],[0,0,1]])
# or equivalent definition
ident_mat = pd.DataFrame(np.identity(3))

# NB: if you try with the following matrix, you find one eigenvalue greater than one and the convergence doesn't work
#dfa = pd.DataFrame([[0,0.9,0.7],[0.5,0,0.7],[0.6,0.7,0]])

eigen_dfa = np.linalg.eig(dfa)[0]
# how many eigenvalues are greater than one?
sum(eigen_dfa > 1)
# check that the matrix identity - dfa is non-singular (meaning that it can be inverted)
np.linalg.det((ident_mat-dfa)) != 0
# or checking that no eigenvalue is = 0
sum(np.linalg.eig((ident_mat-dfa))[0] == 0) == 0
# we take the inverse of identity - dfa
inv_dfa = pd.DataFrame(np.linalg.inv((ident_mat-dfa).values), dfa.columns, dfa.index)
# we create a shock vector, with shocks of 0.5%
chocs = pd.DataFrame([0.005,0.005,0.005])
# compute the shocks impact to infinity
out_loop = inv_dfa.dot(chocs)

# manual loop
# initial shocks values
out_loop_manual = chocs.copy(deep = True)
# initial exposure matrix
dfmult = dfa.copy(deep = True)
for i in range(0,1000):
    out_loop_manual = (dfmult).dot(chocs) + out_loop_manual
    dfmult = dfmult.dot(dfa)

# compare out_loop and out_loop_manual
out_loop - out_loop_manual
# try again with dfa = pd.DataFrame([[0,0.9,0.7],[0.5,0,0.7],[0.6,0.7,0]]), as there is an eigenvalue greater than one this should not work any more

del i, chocs, out_loop, out_loop_manual, ident_mat, inv_dfa, eigen_dfa, dfa, dfmult


#%% Cholesky decomposition


# generate three random variables Z iid N(0,1), sample size 10^5
Nsample = 10**5
N = 3
Z = pd.DataFrame(index=range(Nsample), columns=range(N))

for i in Z.columns:
    Z[i] = np.random.default_rng(i).normal(0, 1, Nsample)

if ploton:
    Z.hist(bins=50)

print('correlations between iid N(0,1) rvs', Z.corr())

# impose a correlation structure
mui = np.log(1)
sigmai = np.sqrt(2*(np.log(2) - mui))
# correlation
rho = .04
# generate the covariance matrix
cov_mat = np.ones([N,N]) * sigmai**2 * rho + np.identity(N) * (sigmai**2 - sigmai**2 * rho)

U = np.linalg.cholesky(cov_mat).T

X = ((U.T)@(Z.T)).T

print('correlations between our rvs', X.corr())
print('standard deviations', X.std())

del cov_mat, i, mui, N, Nsample, rho, sigmai, U, X, Z
