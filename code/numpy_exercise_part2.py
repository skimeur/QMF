#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QMF — Quantitative Methods in Finance
Python for non-programmers: NumPy exercise (Part 2)

This script accompanies Section "Python for non-programmers: numpy exercise part 2"
of the lecture notes *Quantitative Methods in Finance* by Eric Vansteenberghe
(arXiv:2601.12896, SSRN 5178205), developed over more than ten years of teaching at
Université Paris 1 Panthéon-Sorbonne (Master Finance, Technology & Data).

Pedagogical objectives:
- Approximate a function by a limit: exp(x) = lim (1 + x/n)^n, and check exp(i pi) = -1
- Approximate an integral by lower and upper Darboux sums (area of an ellipse)
- Approximate a function by a truncated series: the Mittag-Leffler function E_{alpha,beta}
- Approximate a derivative by a finite difference: ln'(x) = 1/x
- Approximate an integral by the midpoint rule: int_1^2 dx/x = ln(2), and
  sin(x) = int_0^x E_{2,1}(-s^2) ds

Main topics covered:
- Vectorised evaluation of functions on NumPy grids
- Truncated series and where they break down (number of terms versus size of the argument)
- Darboux sums, midpoint rule, forward differences
- scipy.optimize.fminbound, scipy.special.gamma and scipy.special.factorial
- Reproducing the figures of the lecture notes (fig/ folder)

Intended audience:
- Economics and finance students with no prior programming background

Usage:
- Run cell by cell (the "# %%" markers are recognised by Spyder and VS Code) or as a
  script: python numpy_exercise_part2.py
- Set SAVE_FIGURES = True to export the figures of the lecture notes to FIG_DIR

File: numpy_exercise_part2.py
Repository: https://github.com/skimeur/QMF

License: MIT (code)
Year: 2026
Author: Eric Vansteenberghe
"""

import os

import numpy
import matplotlib.pyplot as plt
from scipy.optimize import fminbound
from scipy.special import factorial, gamma

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SHOW_PLOTS = True      # display the figures on screen
SAVE_FIGURES = False   # export the figures of the lecture notes as PDF files
FIG_DIR = "fig"        # export folder, relative to the current working directory


def finish_figure(filename):
    """Export the current figure if SAVE_FIGURES, display it if SHOW_PLOTS, then close it.

    Parameters
    ----------
    filename : str
        Name of the PDF file written in FIG_DIR when SAVE_FIGURES is True.
    """
    if SAVE_FIGURES:
        os.makedirs(FIG_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIG_DIR, filename))
    if SHOW_PLOTS:
        plt.show()
    plt.close()


# --------------------------------------------------------------------------- #
# Functions used in the exercises
# --------------------------------------------------------------------------- #
def exp_approx(x, n):
    """Approximate exp(x) by (1 + x/n)**n, which converges to exp(x) as n -> infinity."""
    return (1 + x / n) ** n


def ellipse_upper(x, a, b):
    """Upper half of the ellipse x^2/a^2 + y^2/b^2 = 1, that is y = b * sqrt(1 - x^2/a^2)."""
    return b * numpy.sqrt(1 - x**2 / a**2)


def darboux_sums(f, lower, upper, n_intervals):
    """Lower and upper Darboux sums of f on [lower, upper].

    The interval is split into n_intervals sub-intervals of equal width. On each
    sub-interval the infimum of f is located numerically with scipy.optimize.fminbound,
    and the supremum as the infimum of -f. Both sums converge to the integral of f when
    f is continuous and n_intervals grows.

    Parameters
    ----------
    f : callable
        Function of one real variable, continuous on [lower, upper].
    lower, upper : float
        Bounds of the integration interval.
    n_intervals : int
        Number of sub-intervals of the partition.

    Returns
    -------
    lower_sum, upper_sum : float
        The lower and upper Darboux sums.
    """
    width = (upper - lower) / n_intervals
    lower_sum = 0.0
    upper_sum = 0.0
    for i in range(n_intervals):
        left = lower + i * width
        right = left + width
        x_inf = fminbound(f, left, right)                   # where f is smallest
        x_sup = fminbound(lambda x: -f(x), left, right)     # where f is largest
        lower_sum += f(x_inf) * width
        upper_sum += f(x_sup) * width
    return lower_sum, upper_sum


def mittag_leffler_11(z, n_terms):
    """Truncated Mittag-Leffler function E_{1,1}(z) = sum_{k=0}^{n_terms-1} z^k / k!.

    E_{1,1} is the exponential function: the sum is its Taylor series, truncated after
    n_terms terms.
    """
    k = numpy.arange(n_terms)
    return numpy.sum(z**k / factorial(k))


def mittag_leffler(z, alpha, beta, n_terms):
    """Truncated generalised Mittag-Leffler function.

    E_{alpha,beta}(z) = sum_{k=0}^{n_terms-1} z^k / Gamma(alpha * k + beta).

    Special cases: E_{1,1}(z) = exp(z), E_{2,1}(-x^2) = cos(x), E_{1,2}(x) = (exp(x) - 1)/x.
    The truncation error grows quickly with |z|: for a given n_terms, the approximation
    is accurate only while |z| is small enough (see the sine exercise below).

    Parameters
    ----------
    z : float or array_like
        Argument(s) of the function; the computation is vectorised over z.
    alpha, beta : float
        Parameters of the function (real parts must be positive).
    n_terms : int
        Number of terms kept in the series.

    Returns
    -------
    float or ndarray
        The truncated series, with the same shape as z.
    """
    z = numpy.asarray(z) * 1.0                 # work in floating point
    k = numpy.arange(n_terms)
    terms = z[..., numpy.newaxis] ** k / gamma(alpha * k + beta)
    return numpy.sum(terms, axis=-1)


def forward_difference(f, x, h):
    """Approximate the derivative f'(x) by the forward difference (f(x + h) - f(x)) / h."""
    return (f(x + h) - f(x)) / h


def midpoint_integral(f, lower, upper, step):
    """Approximate the integral of f on [lower, upper] by the midpoint rule.

    The interval is covered by rectangles of width step whose heights are the values of
    f at the midpoints lower + step/2, lower + 3 step/2, ..., the last midpoint being
    below upper. Negative values of f count as negative areas.
    """
    midpoints = numpy.arange(lower + step / 2, upper, step)
    return numpy.sum(step * f(midpoints))


# %% Approximating the exponential function: exp(x) = lim (1 + x/n)^n

# a first look at the exponential function: exp(x) -> 0 when x -> -infinity
x_check = numpy.arange(-100, 101, 1)
plt.plot(x_check, numpy.exp(x_check))
plt.title("exp(x) on [-100, 100]")
finish_figure("exp_check.pdf")

# compare exp(x) with (1 + x/n)^n for n = 10, 100 and 1000 on [0, 10)
x_grid = numpy.arange(0, 10, 0.01)
y_exp = numpy.exp(x_grid)
plt.plot(x_grid, y_exp, 'k', label="exp(x)")
plt.plot(x_grid, exp_approx(x_grid, 10), 'bs', markersize=2, label="n = 10")
plt.plot(x_grid, exp_approx(x_grid, 100), 'g^', markersize=2, label="n = 100")
plt.plot(x_grid, exp_approx(x_grid, 1000), 'r--', label="n = 1000")
plt.legend()
finish_figure("exp_equa.pdf")

# the total absolute gap over the grid shrinks as n grows (here n = 10, 10^6 + 10, ...)
for n in numpy.arange(10, 11 * 10**6, 10**6):
    total_gap = numpy.sum(numpy.abs(y_exp - exp_approx(x_grid, n)))
    print(f"n = {n:>9d}: sum over the grid of |exp(x) - (1 + x/n)^n| = {total_gap:.4g}")

# Euler's identity exp(i pi) = -1: the points (1 + i pi/n)^n spiral towards -1 in the
# complex plane as n grows
n_grid = numpy.arange(1, 10**4, 0.1)
euler_points = (1 + 1j * numpy.pi / n_grid) ** n_grid
plt.scatter(euler_points.real, euler_points.imag, s=1)
plt.xlabel("real part")
plt.ylabel("imaginary part")
finish_figure("formuledemo.pdf")
print("numpy.exp(1j * numpy.pi) =", numpy.exp(1j * numpy.pi))

# %% Area of an ellipse by lower and upper Darboux sums (Riemann integral)

a = 20      # half width of the ellipse
b = 0.01    # half height of the ellipse

# draw the ellipse x^2/a^2 + y^2/b^2 = 1
x_ellipse = numpy.arange(-a, a, 0.01)
y_ellipse = ellipse_upper(x_ellipse, a, b)
plt.scatter(x_ellipse, y_ellipse, c='blue', s=1)
plt.scatter(x_ellipse, -y_ellipse, c='blue', s=1)
plt.plot([0, 0], [0, b], color='red', linestyle='solid')
plt.plot([0, a], [0, 0], color='red', linestyle='solid')
plt.text(0.1, b / 2, "b")
plt.text(a / 2, 0.0001, "a")
plt.text(-1, -0.001, "(0,0)")
finish_figure("ellipse_a_b.pdf")

# the area of the upper half of the ellipse is the integral of y(x) from -a to a,
# squeezed between the lower and the upper Darboux sums of a partition of [-a, a]
n_intervals = 60
darboux_lower, darboux_upper = darboux_sums(
    lambda x: ellipse_upper(x, a, b), -a, a, n_intervals)
print(f"lower Darboux sum with {n_intervals} intervals: {darboux_lower:.6f}")
print(f"upper Darboux sum with {n_intervals} intervals: {darboux_upper:.6f}")
print(f"exact area of the half ellipse, pi a b / 2   : {numpy.pi * a * b / 2:.6f}")

# %% Mittag-Leffler function E_{1,1} as an approximation of the exponential function

# E_{1,1}(z) = sum z^k / k! is the Taylor series of exp(z); we truncate it after
# 10 and then 15 terms and compare with exp on [0, 10)
y_mittag_10 = [mittag_leffler_11(x, 10) for x in x_grid]
y_mittag_15 = [mittag_leffler_11(x, 15) for x in x_grid]

plt.plot(x_grid, y_exp, 'k', label="exp(x)")
plt.plot(x_grid, y_mittag_10, 'r', label="E_{1,1}, 10 terms")
plt.plot(x_grid, y_mittag_15, 'r--', label="E_{1,1}, 15 terms")
plt.legend()
finish_figure("mittag_exp.pdf")

# %% Approximating a derivative by a finite difference: ln'(x) = 1/x

x_ln = numpy.arange(0.1, 2, 0.01)
one_over_x = 1 / x_ln       # the exact derivative of ln(x)

# the forward difference (ln(x + h) - ln(x)) / h gets closer to 1/x as h shrinks
h_large = 0.3
h_small = 10**(-2)
plt.plot(x_ln, one_over_x, 'k', label="1/x")
plt.plot(x_ln, forward_difference(numpy.log, x_ln, h_large), 'bs', markersize=2,
         label=f"forward difference, h = {h_large}")
plt.plot(x_ln, forward_difference(numpy.log, x_ln, h_small), 'r--',
         label=f"forward difference, h = {h_small}")
plt.legend()
finish_figure("ln_der.pdf")

# %% Approximating an integral by the midpoint rule: int_1^2 dx/x = ln(2)

step_ln = 10**(-3)
area_one_over_x = midpoint_integral(lambda x: 1 / x, 1, 2, step_ln)
print(f"midpoint rule with step {step_ln}: {area_one_over_x:.8f}")
print(f"ln(2)                          : {numpy.log(2):.8f}")

# %% Approximating the sine function with an integral of a Mittag-Leffler function

# sin(x) = int_0^x E_{2,1}(-s^2) ds, because E_{2,1}(-s^2) = cos(s).
# For each x between 0 and 14 we integrate the truncated E_{2,1}(-s^2) with the
# midpoint rule. Two approximations are at play: the width of the rectangles
# (integration step) and the number of terms kept in the series (n_terms).
integration_step = 10**(-2)
x_sine = numpy.arange(0, 15, 0.1)


def sine_from_mittag_leffler(x_values, n_terms):
    """Approximate sin(x) for each x in x_values by int_0^x E_{2,1}(-s^2) ds."""
    return [midpoint_integral(lambda s: mittag_leffler(-s**2, 2, 1, n_terms),
                              0, x, integration_step)
            for x in x_values]


y_sine_17 = sine_from_mittag_leffler(x_sine, n_terms=17)   # the figure of the lecture notes
y_sine_30 = sine_from_mittag_leffler(x_sine, n_terms=30)   # more terms in the series

plt.plot(x_sine, numpy.sin(x_sine), 'k', label="sin(x)")
plt.plot(x_sine, y_sine_17, 'r--', label="integral of E_{2,1}, 17 terms")
plt.plot(x_sine, y_sine_30, 'g:', label="integral of E_{2,1}, 30 terms")
plt.ylim([-1.5, 3])
plt.legend()
finish_figure("mittag_sin.pdf")

# With 17 terms the series E_{2,1}(-s^2) is accurate only for s below about 9 and
# explodes beyond, which the 17-term curve shows past x = 12; with 30 terms the
# approximation holds on the whole interval. The number of terms needed grows with |z|.
print("largest |sin(x) - approximation| on [0, 14] with 17 terms:",
      f"{numpy.max(numpy.abs(numpy.sin(x_sine) - y_sine_17)):.3g}")
print("largest |sin(x) - approximation| on [0, 14] with 30 terms:",
      f"{numpy.max(numpy.abs(numpy.sin(x_sine) - y_sine_30)):.3g}")

# %% Question of the lecture notes: show visually that E_{1,2}(x) = (exp(x) - 1) / x


