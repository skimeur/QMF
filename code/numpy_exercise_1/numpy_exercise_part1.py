#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QMF — Quantitative Methods in Finance
Python for non-programmers: NumPy exercise (Part 1)

This script accompanies Section "Python for non-programmers: numpy exercise part 1"
of the lecture notes *Quantitative Methods in Finance* by Eric Vansteenberghe,
developed over more than ten years of teaching at Université Paris 1 Panthéon-Sorbonne
(Master Finance, Technology & Data).

Pedagogical objectives:
- Introduce NumPy for numerical computation
- Work with vectors and vectorized functions
- Evaluate and plot y = f(x) for scalar and vector inputs
- Define user functions and find roots (manual search, fsolve)
- Illustrate fixed-point iteration and its convergence properties
- Connect numerical methods to Newton’s method
- Introduce gradient descent in one dimension

Main topics covered:
- NumPy arrays and universal functions (exp, cos, etc.)
- Vectorized computation and plotting
- Root finding with scipy.optimize.fsolve
- Fixed-point iteration (Babylonian method, cosine example)
- Banach fixed-point theorem (illustrative, not formal)
- Gradient descent for function minimization

Intended audience:
- Economics and finance students with no prior programming background

File: numpy_exercise_part1.py
Repository: https://github.com/skimeur/QMF

License: MIT (code)
Year: 2026
Author: Eric Vansteenberghe
"""


import os
# change to your directory if you want to be able to export figures in a precise folder
os.chdir('//Users/skimeur/Mon Drive/Musik/piano')

#%% first section, discover basic numpy functions
import numpy

#%% Exponential function

# return exponential of 1.5
numpy.exp(1.5)

# create to arrays, x1 and x2
x1 = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

x2 = numpy.arange(10)

# test that x1 = x2
x1 == x2

# check the sum, true = 1, false = 0
sum(x1 == x2)

# apply the exponential function to all elements of x1
y1 = numpy.exp(x1)

import numpy
import matplotlib.pyplot as plt

# plot y1 as a function of x1
plt.plot(x1, y1)

# if we want a more precise plot
x3 = numpy.arange(0, 10, 0.1)
y3 = numpy.exp(x3)
plt.plot(x3,y3)
#plt.savefig('fig/exponentialplot.pdf')

#%% Define your own function
import numpy
import matplotlib.pyplot as plt

# Define our own function y = x^2 + x -2
def my_f(x):
    return x**2 + x - 2

y_f = my_f(x1)

plt.plot(x1, y_f)

# show
x20 = numpy.arange(-5, 5, .01)
y20 = my_f(x20)
y0 = y20 * 0

plt.plot(x20, y20)
plt.plot(x20, y0)
plt.show()


#%% Plot my function

xsteps = numpy.arange(21)
xsteps = xsteps - 10
ymy_f = my_f(xsteps)
plt.plot(xsteps, ymy_f)

#%% Find roots

import numpy
import matplotlib.pyplot as plt

# Manual search for the root
my_f(0) == 0

my_f(1) == 0

my_f(-2) == 0

# The for loop concept
for element in x1:
    print(element)

# Do a loop over all elements in x1 to search for the root(s)
for element in x1:
    if my_f(element) == 0:
        print('0=f(x) for x=',element)

# Actually there is a function that already exists to search for roots
from scipy.optimize import fsolve
# to get help on fsolve, type cmd + i in front of the line
fsolve(my_f, 20)

my_f(-2) == 0

# define a second function
def second_f(x):
    return numpy.cos(x**2)
    
# find the root of this function
sol = fsolve(second_f,10)

x3 = numpy.arange(-10,10,0.1)
y3 = second_f(x3)
plt.plot(x3,y3)
#plt.savefig('fig/cossquared.pdf')


#%% fixed-point iteration

import numpy
import matplotlib.pyplot as plt

# BABYLONIAN METHOD

def fpi(func, a, x0, nint):
    n = 0
    y = x0
    while n < nint:
        y = func(a, y)
        n += 1
    return y

def babylonian_m(a, x):
    return .5 * (a/x + x)
    
# apply the fixed-point iteration
a = 40 # the number we want the square root of
x0 = 6 # our starting guess
nint = 10 # the number of iteration
print("estimate of sqrt(",a,") with",x0,"as starting guess and", nint,"iterations: ", fpi(babylonian_m, a, x0, nint))
# check with the actual square root
print("the actual sqrt(",a,"):",numpy.sqrt(a))

# BANACH FIXED POINT THEOREM
x0 = 1
nint = 30
# fixed-point iteration
def fcos(x):
    return numpy.cos(x) - x
x_fcos = fsolve(fcos,1)
y = x0
n = 1
q = 0.85
while n < nint:
    y = numpy.cos(y)
    print("we verify the convergence",numpy.abs(y-x_fcos) < numpy.abs(x0 - numpy.cos(x0)) * (q**n) / (1-q))
    n += 1
    
# ANOTHER FUNCTION

#%% Gradient descent
    
import numpy
import matplotlib.pyplot as plt
next_x = 6  # We start the search at x=6
gamma = 0.01  # Step size multiplier
precision = 0.00001  # Desired precision of result
max_iters = 10000  # Maximum number of iterations

# Original function
def f_orig(x):
    return x**4 - 3*x**3 + 2

# Derivative function
def df(x):
    return 4 * x ** 3 - 9 * x ** 2


for _ in range(max_iters):
    current_x = next_x
    next_x = current_x - gamma * df(current_x)

    step = next_x - current_x
    if abs(step) <= precision:
        break

print("Minimum at ", next_x)
print("Minimum is", f_orig(next_x))

# The output for the above will be something like
# "Minimum at 2.2499646074278457"

x4= numpy.arange(-3, 4, 0.1)
y4 = f_orig(x4)
plt.plot(x4,y4)
#plt.savefig('fig/gradientdescent.pdf')

from scipy.optimize import minimize
minimize(f_orig,6)
