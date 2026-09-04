#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantitative Methods in Finance — Beginner exercise with NumPy (part 3)

Topics:
1) Monte Carlo estimation of π via area ratio
2) Sampling distribution of the estimator across repeated runs
3) Monte Carlo approximation of ∫_0^1 e^x dx with error quantification

Notes:
- One RNG (np.random.Generator) is used for full reproducibility.
- Vectorized draws replace Python loops where useful.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd

# ------------------------------ config --------------------------------- #
PLOT = True               # toggle plots
SEED = 77                  # reproducibility
N_POINTS = 10**4        # points for single π run
N_ROUNDS = 100            # repeated runs for sampling distribution
NS_INTEGRAL = 10**5       # points for integral estimation
# ----------------------------------------------------------------------- #

rng = np.random.default_rng(SEED)


# Standard normal pdf
def stdnorm_pdf(x):
    return np.exp(-0.5 * x**2) / np.sqrt(2*np.pi)

# Sample sizes
sample_sizes = [10**1, 10**3, 10**6]

# Theoretical curve grid
xgrid = np.linspace(-4, 4, 400)
pdf_vals = stdnorm_pdf(xgrid)

# Use common bins so histograms are comparable across n
common_bins = np.linspace(-4, 4, 31)  # 30 equal-width bins in [-4,4]

# Draw and store samples once to reuse for plotting and probabilities
samples = {n: rng.standard_normal(n) for n in sample_sizes}

# ---- Plot: all empirical histograms + theoretical N(0,1) pdf ----
plt.figure(figsize=(8, 6))
for n in sample_sizes:
    plt.hist(
        samples[n],
        bins=common_bins,
        density=True,
        alpha=0.35,            # transparency so overlaps are visible
        label=f"n = {n}",
    )

plt.plot(xgrid, pdf_vals, lw=2, label="N(0,1) PDF")
plt.title("Empirical histograms vs. standard normal PDF")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()

# ---- Empirical vs theoretical probability for X < -2 ----
theoretical_p = norm.cdf(-2)  # ≈ 0.0228

records = []
for n in sample_sizes:
    data = samples[n]
    emp_p = (data < -2).mean()
    # Binomial standard error for the empirical probability
    se = np.sqrt(emp_p * (1 - emp_p) / n)
    records.append({
        "n": n,
        "Empirical P(Z<-2)": emp_p,
        "SE": se,
        "Theoretical": theoretical_p,
        "Abs. Error": abs(emp_p - theoretical_p),
    })

df = pd.DataFrame(records)
# Formatted printout
with pd.option_context("display.float_format", "{:.6f}".format):
    print(df.to_string(index=False))

# ----------------------- 1) Monte Carlo for π -------------------------- #
# Draw uniformly on the square (-1,1)×(-1,1)
x = rng.uniform(-1.0, 1.0, N_POINTS)
y = rng.uniform(-1.0, 1.0, N_POINTS)

# Distance from origin and indicator for unit disk
r2 = x*x + y*y
inside = r2 <= 1.0                   # boolean mask
p_hat = inside.mean()                # estimate of P(inside unit circle) = π/4
pi_hat = 4.0 * p_hat                 # π estimator

# Estimated standard error: Var(4*I)/N = 16*p*(1-p)/N; plug-in p̂
se_pi = 4.0 * np.sqrt(p_hat * (1.0 - p_hat) / N_POINTS)
ci95_pi = (pi_hat - 1.96 * se_pi, pi_hat + 1.96 * se_pi)

print(f"[π] MC estimate = {pi_hat:.6f} | numpy π = {np.pi:.6f}")
print(f"[π] SE = {se_pi:.6f} | 95% CI = [{ci95_pi[0]:.6f}, {ci95_pi[1]:.6f}]")

if PLOT:
    # Color points by membership in the disk
    plt.figure()
    plt.scatter(x, y, c=inside.astype(int), s=5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("Uniform draws on (-1,1)×(-1,1): inside unit circle shaded")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.show()

# ------------- 2) Sampling distribution over repeated runs ------------- #
# Vectorized simulation: draw all rounds at once to avoid Python loops
# For memory safety, draw per round; N_ROUNDS×N_POINTS is small here, so vectorize.
x_mat = rng.uniform(-1.0, 1.0, (N_ROUNDS, N_POINTS))
y_mat = rng.uniform(-1.0, 1.0, (N_ROUNDS, N_POINTS))
inside_mat = (x_mat*x_mat + y_mat*y_mat) <= 1.0
p_hat_vec = inside_mat.mean(axis=1)
pi_hat_vec = 4.0 * p_hat_vec

pi_errors = pi_hat_vec - np.pi
mean_est = pi_hat_vec.mean()
std_est = pi_hat_vec.std(ddof=1)

print(f"[π | repeated] Mean(π̂) = {mean_est:.6f} | SD(π̂) = {std_est:.6f}")
print(f"[π | repeated] Mean error = {(mean_est - np.pi):.6f}")

if PLOT:
    plt.figure()
    plt.hist(pi_hat_vec, bins=20, density=True)
    plt.axvline(np.pi, color='red', linestyle='-', linewidth=2, label='True π')  # red vertical line
    plt.title(f"Sampling distribution of π̂ over {N_ROUNDS} runs (N={N_POINTS} each)")
    plt.xlabel("π̂")
    plt.ylabel("Density")
    plt.legend()  # show label in legend
    plt.tight_layout()
    plt.show()


# ------ 3) Monte Carlo approximation of ∫_0^1 e^x dx = e − 1 --------- #
u = rng.uniform(0.0, 1.0, NS_INTEGRAL)
g = np.exp(u)                         # integrand at uniform draws
integ_hat = g.mean()                  # unbiased for E[e^U] = ∫_0^1 e^x dx
se_int = g.std(ddof=1) / np.sqrt(NS_INTEGRAL)
ci95_int = (integ_hat - 1.96 * se_int, integ_hat + 1.96 * se_int)
true_value = np.e - 1.0

print(f"[∫ e^x dx]_MC = {integ_hat:.6f} | true = {true_value:.6f}")
print(f"[∫ e^x dx] SE = {se_int:.6f} | 95% CI = [{ci95_int[0]:.6f}, {ci95_int[1]:.6f}]")
