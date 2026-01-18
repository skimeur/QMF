#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QMF — Quantitative Methods in Finance
Python: variables, functions. An introduction

Companion code for the lecture notes "Quantitative Methods in Finance",
developed by Eric Vansteenberghe over more than ten years of teaching at
Université Paris 1 Panthéon-Sorbonne (Master "Finance, Technology & Data").

Section coverage:
- Built-in immutable types (None, bool, int, float, complex, str, tuple)
- Built-in mutable types (list, dict) and aliasing
- Conditions and loops (if/elif/else, for, while)
- User-defined functions
- Introductory examples with NumPy and pandas
- Notes on object identity and copy semantics (intro level)

File: vansteenberghe_types_loops_functions.py
Repository: https://github.com/skimeur/QMF

License: MIT (code). See LICENSE at repository root.
"""

# ------------------------------------------------------------
# Basic variables and types
# ------------------------------------------------------------

# Define a variable named "growth"
growth = 1
print("growth =", growth, "| type:", type(growth))
# Python infers an integer type (int)

# Reassign the variable with a floating-point value
growth = 1.7
print("growth =", growth, "| type:", type(growth))
# The type is now float

# Extracting the integer part (does NOT modify the variable)
int_part = int(growth)
rounded_part = round(growth)

print("int(growth)   =", int_part)
print("round(growth) =", rounded_part)
print("growth =", growth, "| type still:", type(growth))

# To actually change the type, reassignment is required
growth = int(growth)
print("growth after casting =", growth, "| new type:", type(growth))

# Question for the reader:
# What is the type of 'growth' after casting? Why does it differ from round(growth)?

# ------------------------------------------------------------
# Object identity, immutability, and aliasing (intro)
# ------------------------------------------------------------
# Key idea:
# - `==` compares values
# - `is` compares object identity (same object in memory)
#
# For immutable objects (e.g., int, float, str, tuple), "aliasing" is usually
# not a practical issue because the object cannot be modified in place.
# Reassignment binds the variable name to a *new* object.

x = 8
y = x

print("x =", x, "| id(x) =", id(x))
print("y =", y, "| id(y) =", id(y))
print("x == y ?", x == y)
print("x is y ?", x is y)

# Reassignment: x is now bound to another integer object
x = 300
print("\nAfter reassignment x = 300")
print("x =", x, "| id(x) =", id(x))
print("y =", y, "| id(y) =", id(y))
print("x == y ?", x == y)
print("x is y ?", x is y)

# NOTE:
# You may observe that small integers sometimes share identities due to
# interpreter optimizations (caching/interning). This is an implementation detail:
# never rely on `is` for numeric or string equality.

# Cleanup (optional in scripts; useful in interactive sessions)
del x, y


# ------------------------------------------------------------
# Strings and type compatibility
# ------------------------------------------------------------

# Define a string variable
country = "France"
print("country =", country, "| type:", type(country))

# Python does not allow arithmetic operations between incompatible types
# (e.g. numbers and strings)
try:
    result = growth + country
except TypeError as e:
    print("TypeError:", e)

# Valid operations depend on the type:
# - numbers can be added to numbers
# - strings can be concatenated with strings

# Example: float + int
growth_tplus1 = 2.1
growth_t = 1

sum_numeric = growth_tplus1 + growth_t
print("growth_tplus1 + growth_t =", sum_numeric,
      "| type:", type(sum_numeric))

# Question for the reader:
# Create a variable growthTplus1 = 2.1 and try to add an integer with a float, what happens?
# ------------------------------------------------------------
# Boolean variables from economic comparisons
# ------------------------------------------------------------

# Example: quarterly GDP levels (ECB SDW, introduced manually)
gdp_2019Q1 = 2_886_662.47
gdp_2018Q1 = 2_812_721.20
gdp_2009Q1 = 2_258_741.06
gdp_2008Q1 = 2_354_944.92

# Boolean indicator of growth
growth = gdp_2019Q1 > gdp_2018Q1
print("Growth 2018Q1 → 2019Q1 ?", growth, "| type:", type(growth))

growth = gdp_2009Q1 > gdp_2008Q1
print("Growth 2008Q1 → 2009Q1 ?", growth)


# ------------------------------------------------------------
# Lists for time-series data
# ------------------------------------------------------------

quarters = ["q2008Q1", "q2009Q1", "q2018Q1", "q2019Q1"]
gdps = [2_354_944.92, 2_258_741.06, 2_812_721.20, 2_886_662.47]

# Lists are mutable
gdps.append(5)
gdps.remove(5)      # or gdps.pop() if the element is last

# Python indexing starts at 0
i = 0
print("First observation:", quarters[i], gdps[i])

# Growth comparison using list indexing
print("GDP growth 2008Q1 → 2009Q1 ?", gdps[1] > gdps[0])


# ------------------------------------------------------------
# Aliasing and mutable objects (lists)
# ------------------------------------------------------------
# Key idea:
# Assigning a list to another variable creates an alias (same object in memory).
# Modifying the list through one name affects all aliases.

gdps_alias = gdps
print("Same object?", gdps_alias is gdps)

# In-place modification affects both references
gdps.append(5)
print("Alias after append:", gdps_alias)

gdps.pop()
print("Alias after pop   :", gdps_alias)

# Creating an independent copy breaks the alias
gdps_copy = gdps.copy()      # equivalent to: gdps[:] or list(gdps)
print("Copy is same object?", gdps_copy is gdps)

gdps.append(5)
print("Original list:", gdps)
print("Copied list  :", gdps_copy)

gdps.pop()


# ------------------------------------------------------------
# Dictionaries: mapping dates to values
# ------------------------------------------------------------
# Lists 'quarters' and 'gdps' are independent objects.
# A dictionary allows us to explicitly link each date to its GDP value.

gdp_by_quarter = dict(zip(quarters, gdps))

# Equivalent explicit construction (for illustration)
gdp_by_quarter_manual = {
    "q2019Q1": 2_886_662.47,
    "q2018Q1": 2_812_721.20,
    "q2009Q1": 2_258_741.06,
    "q2008Q1": 2_354_944.92,
}

# Growth comparison using dictionary keys
print(
    "GDP growth 2008Q1 → 2009Q1 ?",
    gdp_by_quarter["q2009Q1"] > gdp_by_quarter["q2008Q1"]
)

# ------------------------------------------------------------
# Conditional statements (if / elif / else)
# ------------------------------------------------------------

gdp_2009 = gdp_by_quarter["q2009Q1"]
gdp_2008 = gdp_by_quarter["q2008Q1"]

if gdp_2009 > gdp_2008:
    outcome = "grew"
elif gdp_2009 == gdp_2008:
    outcome = "stagnated"
else:
    outcome = "fell"

print(f"GDP in Europe between 2008Q1 and 2009Q1 {outcome}.")

# ------------------------------------------------------------
# Loops: for and while
# ------------------------------------------------------------

# for-loop over indices
for i in range(len(gdps)):
    print("Index:", i)

# for-loop over consecutive observations
for i in range(len(gdps) - 1):
    change = gdps[i + 1] - gdps[i]
    print(f"Between {quarters[i]} and {quarters[i + 1]}, GDP changed by {change:,.2f}")

# while-loop: detect when a condition becomes true
# Example: first date when GDP exceeds a threshold
threshold = 2_400_000
i = 0

while gdps[i] < threshold:
    i += 1

print(f"GDP crossed {threshold:,.0f} between {quarters[i - 1]} and {quarters[i]}")


# ------------------------------------------------------------
# Functions: reuse computations
# ------------------------------------------------------------

def growth_rate(series, i, j):
    """
    Simple growth rate between two observations in a level series.

    Parameters
    ----------
    series : list[float] or array-like
        Level series (e.g., GDP levels).
    i, j : int
        Indices of the initial and final observations.

    Returns
    -------
    float
        (series[j] - series[i]) / series[i]
    """
    return (series[j] - series[i]) / series[i]


# Same computation "inline" (repetition)
for i in range(len(gdps) - 1):
    g = (gdps[i + 1] - gdps[i]) / gdps[i]
    print(f"Growth {quarters[i]} → {quarters[i + 1]}: {g:.4%}")

# Same computation via a function (cleaner, reusable)
for i in range(len(gdps) - 1):
    g = growth_rate(gdps, i, i + 1)
    print(f"Growth {quarters[i]} → {quarters[i + 1]}: {g:.4%}")

print("GDP growth 2008Q1 → 2009Q1:", f"{growth_rate(gdps, 0, 1):.4%}")


# ------------------------------------------------------------
# Growth rates: algebra vs numerical precision
# ------------------------------------------------------------
# In theory:
#   (x - y) / y  ==  x / y - 1
# In practice:
#   floating-point arithmetic introduces small rounding errors.

x = gdps[1]
y = gdps[0]

# Two equivalent formulas (in exact arithmetic)
d1 = (x - y) / y          # definition of growth rate
d2 = x / y - 1            # algebraically equivalent

print("d1 =", d1)
print("d2 =", d2)
print("d1 == d2 ?", d1 == d2)

# Log-growth as a common approximation in economics
import numpy as np
log_growth = np.log(x) - np.log(y)
print("log-growth approximation =", log_growth)

# High-precision arithmetic to assess numerical accuracy
from decimal import Decimal, getcontext
getcontext().prec = 30

xD = Decimal(x)
yD = Decimal(y)
d_exact = (xD - yD) / yD

# Compare floating-point errors
err_d1 = abs(Decimal(d1) - d_exact)
err_d2 = abs(Decimal(d2) - d_exact)

print("Error using (x - y) / y :", err_d1)
print("Error using x / y - 1   :", err_d2)
print("Is (x - y) / y more accurate?", err_d1 < err_d2)

# Takeaway:
# Prefer (x - y) / y for numerical stability in finite-precision arithmetic.


# ------------------------------------------------------------
# Functions returning multiple outputs
# ------------------------------------------------------------

def growth_stats(series, i, j):
    """
    Return both the growth rate and the level difference
    between two observations in a series.

    Returns
    -------
    growth_rate : float
        (series[j] - series[i]) / series[i]
    level_diff : float
        series[j] - series[i]
    """
    growth_rate = (series[j] - series[i]) / series[i]
    level_diff = series[j] - series[i]
    return growth_rate, level_diff


# Tuple unpacking
g_rate, g_diff = growth_stats(gdps, 0, 1)

print(
    f"GDP growth 2008Q1 → 2009Q1: {g_rate:.4%}, "
    f"level change: {g_diff:,.2f}"
)


# ------------------------------------------------------------
# NumPy arrays
# ------------------------------------------------------------

import numpy as np

# GDP levels as a NumPy array (efficient numerical operations)
gdps_np = np.array([2_886_662.47, 2_812_721.20, 2_258_741.06, 2_354_944.92])

# Element-wise comparison
print(
    "GDP growth 2008Q1 → 2009Q1 ?",
    gdps_np[2] > gdps_np[3]
)

# ------------------------------------------------------------
# pandas DataFrame: labeled data
# ------------------------------------------------------------

import pandas as pd

# Create a DataFrame with labeled rows and columns
df = pd.DataFrame(
    {"GDP": [2_886_662.47, 2_812_721.20, 2_258_741.06, 2_354_944.92]},
    index=["q2019Q1", "q2018Q1", "q2009Q1", "q2008Q1"]
)

# Label-based access with .loc
print(
    "GDP growth 2008Q1 → 2009Q1 ?",
    df.loc["q2009Q1", "GDP"] > df.loc["q2008Q1", "GDP"]
)

# Column access (two equivalent syntaxes)
df["GDP"]
df.GDP

# Conditional selection
threshold = 2_400_000
dates_above = df.loc[df.GDP > threshold].index
print("GDP above threshold at:", list(dates_above))

# Iteration (usually discouraged in pandas, shown for illustration)
for date, gdp in df.GDP.items():
    if gdp > threshold:
        print(f"GDP was above {threshold:,.0f} at date {date}")

# Key takeaway:
# Prefer vectorized operations (df.loc[condition]) over explicit loops in pandas.

# ------------------------------------------------------------
# Subtlety: mutable objects inside a DataFrame cell
# ------------------------------------------------------------
# Even with df.copy(deep=True), pandas does not deep-copy *Python objects*
# stored inside cells (like lists). This can create surprising aliasing.

gdp_list = [2_354_944.92, 2_258_741.06, 2_812_721.20, 2_886_662.47]

df2 = pd.DataFrame({0: [gdp_list]})          # one row, one column, a list stored in the cell
df2_copy = df2.copy(deep=True)

print("Same DataFrame object?", df2_copy is df2)
print("Cell object shared?   ", df2_copy.loc[0, 0] is df2.loc[0, 0])

# Mutate the list in place (inside the DataFrame cell)
df2.loc[0, 0].append(4)

print("Original DataFrame cell:", df2.loc[0, 0])
print("Copied DataFrame cell  :", df2_copy.loc[0, 0])

# Takeaway:
# Avoid storing mutable objects (lists, dicts) inside DataFrame cells.
# Prefer "long" tables: one value per cell / row, with explicit columns.


        
# ------------------------------------------------------------
# (Mini) performance comparison: NumPy vs pandas
# ------------------------------------------------------------
# Works in a standard Python script (no IPython magics).

import timeit
import numpy as np
import pandas as pd
import random
import scipy.sparse

# --- 1) Indexing and elementwise multiplication (1D) ---
a = np.arange(100)
b = np.arange(100, 200)

s = pd.Series(a)
t = pd.Series(b)

idx = np.random.choice(a, size=10, replace=False)

n = 50_000  # number of repetitions (adjust if too slow)

np_index_time = timeit.timeit(lambda: a[idx], number=n)
pd_index_time = timeit.timeit(lambda: s.iloc[idx], number=n)

np_mul_time = timeit.timeit(lambda: a * b, number=n)
pd_mul_time = timeit.timeit(lambda: s * t, number=n)

print("\nIndexing (smaller is faster)")
print(f"NumPy a[idx]     : {np_index_time:.4f}s for n={n}")
print(f"pandas s.iloc[idx]: {pd_index_time:.4f}s for n={n}")

print("\nElementwise multiplication")
print(f"NumPy a*b        : {np_mul_time:.4f}s for n={n}")
print(f"pandas s*t       : {pd_mul_time:.4f}s for n={n}")

# --- 2) Matrix-vector multiplication (2D) ---
density = random.uniform(0.1, 0.2)
M = scipy.sparse.rand(200, 200, density=density, format="csr").toarray()
v = np.random.randn(200)

M_pd = pd.DataFrame(M)
v_pd = pd.Series(v)

m = 5_000  # repetitions (matrix mult is heavier)

np_mv_time = timeit.timeit(lambda: M @ v, number=m)
pd_mv_time = timeit.timeit(lambda: M_pd.to_numpy() @ v_pd.to_numpy(), number=m)

print("\nMatrix-vector multiplication")
print(f"NumPy M@v                      : {np_mv_time:.4f}s for m={m}")
print(f"pandas -> NumPy -> (M@v)        : {pd_mv_time:.4f}s for m={m}")

print("\nTakeaway: use pandas for labeled data, NumPy for heavy linear algebra.")
