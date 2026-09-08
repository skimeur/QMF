#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Eric Vansteenberghe
Quantitative Methods in Finance
Beginner exercise with pandas DataFrames - part 1
Computational Complexity
2024
"""

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% Naive matrix multiplication with pandas .dot()
# inspired from https://stackoverflow.com/questions/51715082/what-is-the-running-time-big-o-order-of-pandas-dataframe-join

def work(n, runs=10**3):
    # Create two random square matrices using pandas DataFrame
    matrix_a = pd.DataFrame(np.random.rand(n, n))
    matrix_b = pd.DataFrame(np.random.rand(n, n))
    
    # List to store timings
    timings = []
    
    # Run multiple times and average the results
    for _ in range(runs):
        t0 = time.time()
        result_matrix = matrix_a.dot(matrix_b)
        dt = time.time() - t0
        timings.append(dt)
    
    # Return average time
    return np.mean(timings)

sizes = (2**np.arange(2, 10)).astype(int)
times = []

for n in sizes:
    dt = work(n)
    times.append(dt)
    print(f'{n}: {dt:.4f}s')

# Fit the times to a cubic model since matrix multiplication is O(n^3)
n = np.array(sizes)
t = np.array(times)
coefficients = np.polyfit(n**3, t, 1)  # Using cubic terms for fitting
model = np.poly1d(coefficients)

# Plotting the results
plt.plot(n, t, 'o', label='Observed Times')
plt.plot(n, model(n**3), '-', label='Fitted Model n^3')
plt.xlabel('Matrix size n')
plt.ylabel('Time (seconds)')
plt.legend()
plt.show()

#%% With the Strassen algorithm
# inspired from https://www.geeksforgeeks.org/strassen-algorithm-in-python/

def strassen(A, B):
    n = len(A)
    
    if n <= 2:  # Base case
        return np.dot(A, B)
    
    # Partition matrices into submatrices
    mid = n // 2
    A11 = A[:mid, :mid]
    A12 = A[:mid, mid:]
    A21 = A[mid:, :mid]
    A22 = A[mid:, mid:]
    B11 = B[:mid, :mid]
    B12 = B[:mid, mid:]
    B21 = B[mid:, :mid]
    B22 = B[mid:, mid:]
    
    # Recursive multiplication
    P1 = strassen(A11, B12 - B22)
    P2 = strassen(A11 + A12, B22)
    P3 = strassen(A21 + A22, B11)
    P4 = strassen(A22, B21 - B11)
    P5 = strassen(A11 + A22, B11 + B22)
    P6 = strassen(A12 - A22, B21 + B22)
    P7 = strassen(A11 - A21, B11 + B12)
    
    # Combine results to form C
    C11 = P5 + P4 - P2 + P6
    C12 = P1 + P2
    C21 = P3 + P4
    C22 = P5 + P1 - P3 - P7
    
    # Combine quadrants to form C
    C = np.vstack((np.hstack((C11, C12)), np.hstack((C21, C22))))
    return C

def work(n, runs=10**3):
    size = 2**np.ceil(np.log2(n)).astype(int)
    timings = []
    
    for _ in range(runs):
        matrix_a = pd.DataFrame(np.random.rand(size, size))
        matrix_b = pd.DataFrame(np.random.rand(size, size))
        t0 = time.time()
        result_matrix = strassen(matrix_a.values, matrix_b.values)
        dt = time.time() - t0
        timings.append(dt)
    
    # Return average time
    return np.mean(timings)

sizes = (2**np.arange(2, 6)).astype(int)
times = []

for n in sizes:
    dt = work(n)
    times.append(dt)
    print(f'{n}: {dt:.4f}s')

# Plotting the results
n = np.array(sizes)
t = np.array(times)
coefficients = np.polyfit(n**np.log2(7), t, 1)
model = np.poly1d(coefficients)

plt.plot(n, t, 'o', label='Observed Times')
plt.plot(n, model(n**np.log2(7)), '-', label='Fitted Model $n^{2.807}$')
plt.xlabel('Matrix size n')
plt.ylabel('Time (seconds)')
plt.legend()
plt.show()
