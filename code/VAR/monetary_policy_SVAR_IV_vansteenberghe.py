#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monetary Policy Shocks: from Stationarity to Causal Inference
==============================================================

Author : Eric Vansteenberghe
Course : Quantitative Methods in Finance (QMF)
License: MIT

Overview
--------
This script is a *self-contained wrap-up exercise* that walks through the
full empirical pipeline used in modern monetary-policy research:

    Simulated data  -->  Stationarity  -->  VAR
      -->  Granger causality  -->  SVAR (Cholesky)  -->  FEVD
      -->  External instruments (LP-IV / SVAR-IV)
      -->  What goes wrong with the wrong ordering
      -->  Non-recursive DGP  -->  Contaminated instrument
      -->  Monte Carlo repetitions

Everything is built from a **simulated New-Keynesian DGP** with known
structural shocks, so students can compare estimates against the truth.

Data Generating Process  (3-equation New-Keynesian model)
---------------------------------------------------------
    g_t  = rho_g * g_{t-1}  -  alpha_r * r_{t-1}  +  eps^d_t   (IS curve)
    pi_t = rho_pi * pi_{t-1} +  kappa * g_{t-1}   +  eps^s_t   (Phillips curve)
    i_t  = phi_pi * pi_t     +  phi_g * g_t        +  eps^m_t   (Taylor rule)

where:
    g_t   = output gap (deviation of GDP from potential)
    pi_t  = inflation rate
    i_t   = nominal interest rate (central bank policy rate)
    r_t   = i_t - pi_t  = ex-post real interest rate

Structural shocks (all mutually independent):
    eps^d_t  ~ N(0, sigma_d^2)   demand shock
    eps^s_t  ~ N(0, sigma_s^2)   supply / cost-push shock
    eps^m_t  ~ N(0, sigma_m^2)   monetary policy shock

Ordering (g, pi, i) is *recursive by construction*: output and prices are
predetermined within the period, while the central bank sets the rate
after observing current output and inflation. This is the Christiano-
Eichenbaum-Evans (1999, 2005) ordering.

External instrument
-------------------
We construct a noisy proxy for the monetary-policy shock:

    z_t  =  eps^m_t  +  eta_t ,   eta_t ~ N(0, sigma_eta^2)

This mimics a high-frequency futures-based surprise (Kuttner 2001) that
is correlated with the true policy shock but contaminated by noise.

Sections
--------
 1. DGP and data simulation
 2. Stationarity: ADF tests (I(0) returns vs I(1) levels)
 3. Reduced-form VAR: lag selection, estimation, diagnostics
 4. Granger causality tests
 5. Structural VAR: Cholesky identification and IRFs (with bootstrap CIs)
 6. Forecast Error Variance Decomposition (FEVD)
 7. True IRFs (known from DGP) --- benchmark
 8. External instruments: LP-IV and SVAR-IV
 9. Comparison: true IRF vs Cholesky-SVAR vs LP-IV vs SVAR-IV
10. What goes wrong with the wrong Cholesky ordering
11. Non-recursive DGP: when Cholesky fails but IV succeeds
12. Contaminated instrument: what happens when exogeneity fails
13. Monte Carlo repetitions: sampling distributions of IRF estimates
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend: no plot windows
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.api import VAR, SVAR
from numpy.linalg import lstsq

plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

os.chdir('/Users/skimeur/Mon Drive/QMF/')

# ── Output directory for figures ────────────────────────────────────────
FIG_DIR = os.path.join('/Users/skimeur/Mon Drive/QMF/fig')
os.makedirs(FIG_DIR, exist_ok=True)

def savefig(name, fig=None):
    """Save the current (or given) figure to the fig/ directory."""
    path = os.path.join(FIG_DIR, name)
    if fig is not None:
        fig.savefig(path, bbox_inches='tight', dpi=150)
    else:
        plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close('all')
    print(f"  [saved] {path}")


# =====================================================================
# 1.  DATA GENERATING PROCESS
# =====================================================================

print("=" * 70)
print("1. DATA GENERATING PROCESS -- 3-equation New-Keynesian model")
print("=" * 70)

# -- Structural parameters -------------------------------------------
T = 500                # sample size (quarters)
rho_g = 0.80           # IS curve: output gap persistence
alpha_r = 0.50         # IS curve: sensitivity to real rate
rho_pi = 0.70          # Phillips curve: inflation persistence
kappa = 0.30           # Phillips curve: slope (output gap -> inflation)
phi_pi = 1.50          # Taylor rule: response to inflation (>1 for determinacy)
phi_g = 0.50           # Taylor rule: response to output gap
sigma_d = 0.50         # std of demand shock
sigma_s = 0.30         # std of cost-push shock
sigma_m = 0.25         # std of monetary-policy shock
sigma_eta = 0.30       # std of instrument noise

np.random.seed(42)

# -- Draw structural shocks (known ground truth) ---------------------
eps_d = np.random.normal(0, sigma_d, T)   # demand shock
eps_s = np.random.normal(0, sigma_s, T)   # supply / cost-push shock
eps_m = np.random.normal(0, sigma_m, T)   # monetary policy shock
eta   = np.random.normal(0, sigma_eta, T) # instrument noise

# -- Simulate the economy -------------------------------------------
g      = np.zeros(T)   # output gap
pi     = np.zeros(T)   # inflation
i_rate = np.zeros(T)   # nominal interest rate
r      = np.zeros(T)   # real rate

for t in range(1, T):
    g[t]      = rho_g * g[t-1] - alpha_r * r[t-1] + eps_d[t]       # IS
    pi[t]     = rho_pi * pi[t-1] + kappa * g[t-1] + eps_s[t]        # Phillips
    i_rate[t] = phi_pi * pi[t] + phi_g * g[t] + eps_m[t]            # Taylor
    r[t]      = i_rate[t] - pi[t]                                    # real rate

# -- External instrument (noisy proxy for monetary shock) ------------
z = eps_m + eta   # correlated with eps^m, orthogonal to eps^d and eps^s

# -- Construct cumulated levels (for I(1) illustration) ---------------
gdp_level   = 100 + np.cumsum(g)
price_level = 100 + np.cumsum(pi)

# -- Assemble DataFrame ----------------------------------------------
data = pd.DataFrame({
    'output_gap':    g,
    'inflation':     pi,
    'interest_rate': i_rate,
    'real_rate':     r,
    'gdp_level':     gdp_level,
    'price_level':   price_level,
    'z_instrument':  z,
})
data.index = pd.date_range('2000-01-01', periods=T, freq='QE')
data.index.name = 'date'

print(f"Sample: {data.index[0].date()} -- {data.index[-1].date()} ({T} obs)")
print("\nTrue structural parameters:")
print(f"  IS curve:      rho_g={rho_g}, alpha_r={alpha_r}")
print(f"  Phillips:      rho_pi={rho_pi}, kappa={kappa}")
print(f"  Taylor rule:   phi_pi={phi_pi}, phi_g={phi_g}")
print(f"  Shock std:     sigma_d={sigma_d}, sigma_s={sigma_s}, sigma_m={sigma_m}")
print(f"  Instrument:    sigma_eta={sigma_eta}")
R2_theory = sigma_m**2 / (sigma_m**2 + sigma_eta**2)
print(f"  First-stage R-squared (theoretical): {R2_theory:.2f}")

# -- Figure: simulated economy ---------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Simulated New-Keynesian Economy", fontsize=13)
for ax, col, lbl in zip(axes.flat,
        ['output_gap', 'inflation', 'interest_rate', 'real_rate'],
        ['Output gap $g_t$', 'Inflation $\\pi_t$',
         'Nominal rate $i_t$', 'Real rate $r_t$']):
    ax.plot(data.index, data[col], linewidth=0.8)
    ax.axhline(0, color='grey', linewidth=0.5, linestyle='--')
    ax.set_title(lbl)
    ax.set_xlabel('')
plt.tight_layout()
savefig('wrapup_simulated_economy.pdf')


# =====================================================================
# 2.  STATIONARITY -- ADF TESTS
# =====================================================================

print("\n" + "=" * 70)
print("2. STATIONARITY -- Augmented Dickey-Fuller tests")
print("=" * 70)

def adf_report(series, name, regression='c'):
    """Run ADF test and print a one-line summary."""
    result = adfuller(series.dropna(), regression=regression, autolag='AIC')
    stat, pval, lags = result[0], result[1], result[2]
    verdict = ("REJECT H0 (stationary)" if pval < 0.05
               else "FAIL to reject H0 (unit root)")
    print(f"  {name:25s}  ADF={stat:+.3f}  p={pval:.4f}  lags={lags}"
          f"  -> {verdict}")
    return pval

print("\n-- I(0) variables (output gap, inflation, interest rate): --")
for col in ['output_gap', 'inflation', 'interest_rate']:
    adf_report(data[col], col)

print("\n-- I(1) variables (cumulated levels): --")
for col in ['gdp_level', 'price_level']:
    adf_report(data[col], col + " (level)")

print("\n-- First differences of levels (should be I(0)): --")
for col in ['gdp_level', 'price_level']:
    adf_report(data[col].diff().dropna(), f"Delta({col})")

print("""
Key lesson:
  * The NK variables (g, pi, i) are stationary -> work directly in a VAR.
  * Cumulated levels (GDP, prices) are I(1) -> need differencing or VECM.
  * First differences of levels recover the original stationary series.
""")


# =====================================================================
# 3.  REDUCED-FORM VAR
# =====================================================================

print("=" * 70)
print("3. REDUCED-FORM VAR -- lag selection and estimation")
print("=" * 70)

# Work with the three stationary macro variables
var_data = data[['output_gap', 'inflation', 'interest_rate']].copy()

# -- Lag selection ---------------------------------------------------
model_sel = VAR(var_data)
lag_table = model_sel.select_order(maxlags=8)
print(lag_table.summary())
p_opt = max(lag_table.aic, 1)  # at least lag 1
print(f"\nSelected lag (AIC): p = {p_opt}")

# -- Estimation ------------------------------------------------------
var_model  = VAR(var_data)
var_results = var_model.fit(p_opt, trend='c')
print(var_results.summary())

# -- Diagnostics -----------------------------------------------------
print("\n-- Diagnostics --")
print(f"Stable: {var_results.is_stable()}")
print(var_results.test_normality())
print(var_results.test_whiteness())


# =====================================================================
# 4.  GRANGER CAUSALITY TESTS
# =====================================================================

print("\n" + "=" * 70)
print("4. GRANGER CAUSALITY TESTS")
print("=" * 70)
print("""
The Granger causality test asks: does knowing the PAST of variable X
improve the forecast of variable Y, beyond Y's own past?
  H0: X does NOT Granger-cause Y  (i.e. lags of X are jointly zero)
  H1: X Granger-causes Y
This is a test of PREDICTIVE content, not structural causality.
""")

gc_pairs = [
    ('output_gap',    'inflation',     "g -> pi (Phillips channel)"),
    ('inflation',     'output_gap',    "pi -> g (real rate channel)"),
    ('output_gap',    'interest_rate', "g -> i  (Taylor rule)"),
    ('inflation',     'interest_rate', "pi -> i (Taylor rule)"),
    ('interest_rate', 'output_gap',    "i -> g  (IS channel)"),
    ('interest_rate', 'inflation',     "i -> pi (indirect)"),
]

for x_col, y_col, label in gc_pairs:
    gc_data = var_data[[y_col, x_col]].dropna()
    print(f"\n  {label}")
    try:
        gc_result = grangercausalitytests(gc_data, maxlag=p_opt, verbose=False)
        for lag, res in gc_result.items():
            ftest = res[0]['ssr_ftest']
            print(f"    lag={lag}: F={ftest[0]:.2f}, p={ftest[1]:.4f}"
                  f"  {'***' if ftest[1]<0.01 else '**' if ftest[1]<0.05 else '*' if ftest[1]<0.10 else ''}")
    except Exception as e:
        print(f"    Error: {e}")

print("""
Key lesson:
  * Granger causality captures predictive linkages, not causal mechanisms.
  * We expect: g -> pi (Phillips), {g, pi} -> i (Taylor rule),
    and i -> g (with a lag, via the real rate / IS curve).
  * Finding that i Granger-causes g does NOT mean we identified a
    monetary policy shock -- the central bank is reacting endogenously.
    This is precisely why we need SVAR or external instruments.
""")


# =====================================================================
# 5.  STRUCTURAL VAR -- Cholesky identification
# =====================================================================

print("=" * 70)
print("5. STRUCTURAL VAR -- Cholesky (short-run) identification")
print("=" * 70)

# Ordering: (output_gap, inflation, interest_rate)
# This is the Christiano-Eichenbaum-Evans ordering:
#   * Output gap is predetermined (does not respond to pi or i on impact)
#   * Inflation is predetermined (does not respond to i on impact)
#   * Interest rate responds to both g and pi contemporaneously (Taylor rule)
#
# A matrix (lower-triangular with 1s on diagonal):
#   A = [[1,   0,   0],
#        ['E', 1,   0],
#        ['E', 'E', 1]]

A_mat = np.array([
    [1,   0,   0],
    ['E', 1,   0],
    ['E', 'E', 1]
])

svar_model   = SVAR(var_data, svar_type='A', A=A_mat)
svar_results = svar_model.fit(maxlags=p_opt, trend='c', solver='nm')

print("\nEstimated A matrix:")
print(np.array2string(svar_results.A, precision=4, suppress_small=True))
print("\nA^{-1} (structural impact matrix S):")
S_hat = np.linalg.inv(svar_results.A)
print(np.array2string(S_hat, precision=4, suppress_small=True))

# -- IRFs from SVAR --------------------------------------------------
n_ahead = 20
irf_svar = svar_results.irf(periods=n_ahead)

fig_irf = irf_svar.plot(orth=False)
fig_irf.suptitle("SVAR IRFs -- Cholesky ordering (g, pi, i)", fontsize=12)
plt.tight_layout()
savefig('wrapup_SVAR_cholesky_IRFs.pdf', fig_irf)

# -- Extract the monetary policy shock IRF (shock 3 = interest_rate) --
# irf_svar.irfs has shape (n_ahead+1, n_vars, n_shocks)
irf_cholesky_g  = irf_svar.irfs[:, 0, 2]  # output gap response to i shock
irf_cholesky_pi = irf_svar.irfs[:, 1, 2]  # inflation response to i shock
irf_cholesky_i  = irf_svar.irfs[:, 2, 2]  # interest rate own response

print("\nMonetary policy shock -> Output gap (Cholesky SVAR):")
print(f"  Impact (h=0): {irf_cholesky_g[0]:+.4f}")
print(f"  h=4:          {irf_cholesky_g[min(4, n_ahead)]:+.4f}")

# -- Bootstrap confidence intervals for Cholesky SVAR IRFs -----------
print("\n-- Bootstrap confidence intervals (Cholesky SVAR) --")
n_boot = 500
boot_irfs = np.zeros((n_boot, n_ahead + 1, 3))  # store monetary shock IRFs

for b in range(n_boot):
    # Resample residuals with replacement
    resid_orig = var_results.resid.values  # (T-p, 3)
    idx_boot = np.random.choice(len(resid_orig), size=len(resid_orig), replace=True)
    resid_boot = resid_orig[idx_boot]

    # Reconstruct data from VAR coefficients + resampled residuals
    coefs = var_results.coefs          # (p, 3, 3)
    intercept = var_results.coefs_exog.flatten()  # (3,) intercept
    y_init = var_data.values[:p_opt]   # initial p observations
    y_boot = np.zeros((len(resid_orig) + p_opt, 3))
    y_boot[:p_opt] = y_init
    for t in range(len(resid_orig)):
        y_t = intercept.copy()
        for lag in range(p_opt):
            y_t += coefs[lag] @ y_boot[p_opt + t - 1 - lag]
        y_t += resid_boot[t]
        y_boot[p_opt + t] = y_t

    # Re-estimate VAR and compute Cholesky IRFs
    try:
        boot_var = VAR(pd.DataFrame(y_boot, columns=var_data.columns))
        boot_res = boot_var.fit(p_opt, trend='c')
        Omega_boot = np.array(boot_res.sigma_u)
        P_boot = np.linalg.cholesky(Omega_boot)
        ma_boot = boot_res.ma_rep(maxn=n_ahead)
        for h in range(n_ahead + 1):
            struct_irf_h = ma_boot[h] @ P_boot
            boot_irfs[b, h, :] = struct_irf_h[:, 2]  # monetary shock column
    except Exception:
        boot_irfs[b] = np.nan

# Remove failed replications
boot_irfs = boot_irfs[~np.isnan(boot_irfs[:, 0, 0])]
print(f"  Successful bootstrap replications: {len(boot_irfs)}/{n_boot}")

# Normalise each bootstrap draw by its own impact response of i
boot_scale = boot_irfs[:, 0, 2]  # (n_valid,)
boot_scale = np.where(np.abs(boot_scale) > 1e-10, boot_scale, 1.0)
boot_irfs_norm = boot_irfs / boot_scale[:, None, None]

# 90% confidence bands
boot_ci_g_lo  = np.percentile(boot_irfs_norm[:, :, 0], 5, axis=0)
boot_ci_g_hi  = np.percentile(boot_irfs_norm[:, :, 0], 95, axis=0)
boot_ci_pi_lo = np.percentile(boot_irfs_norm[:, :, 1], 5, axis=0)
boot_ci_pi_hi = np.percentile(boot_irfs_norm[:, :, 1], 95, axis=0)

print(f"  Output gap 90% CI at h=4: [{boot_ci_g_lo[4]:+.4f}, {boot_ci_g_hi[4]:+.4f}]")


# =====================================================================
# 6.  FORECAST ERROR VARIANCE DECOMPOSITION (FEVD)
# =====================================================================

print("\n" + "=" * 70)
print("6. FORECAST ERROR VARIANCE DECOMPOSITION (FEVD)")
print("=" * 70)
print("""
The FEVD tells us what fraction of the h-step-ahead forecast error
variance of each variable is attributable to each structural shock.
""")

# We compute the FEVD manually from the structural IRFs, which is
# more pedagogical than calling a black-box function.
#
# At horizon h, the forecast error variance of variable i due to shock j:
#   FEVD_{i,j}(h) = [sum_{k=0}^{h} psi_{i,j}^k ^2]
#                  / [sum_{k=0}^{h} sum_{l=1}^{n} psi_{i,l}^k ^2]
#
# where psi_{i,j}^k is the structural IRF of variable i to shock j at lag k.

n_vars = 3
irf_array = irf_svar.irfs   # shape (n_ahead+1, n_vars, n_shocks)

# Cumulative squared contributions: numerator
cum_sq = np.cumsum(irf_array ** 2, axis=0)   # (n_ahead+1, n_vars, n_shocks)

# Total forecast error variance at each horizon: denominator
total_fev = cum_sq.sum(axis=2, keepdims=True)  # (n_ahead+1, n_vars, 1)

# FEVD: fraction of variance explained
fevd_array = cum_sq / np.where(total_fev > 0, total_fev, 1.0)  # avoid /0

# -- FEVD figure (stacked bar chart) ---------------------------------
h_range = np.arange(n_ahead + 1)
fig_fevd, axes_fevd = plt.subplots(1, 3, figsize=(15, 5))
fig_fevd.suptitle("Forecast Error Variance Decomposition (Cholesky ordering)",
                   fontsize=13)
var_names_short = ['output_gap', 'inflation', 'interest_rate']
shock_colors = ['#2166ac', '#b2182b', '#4daf4a']
shock_labels = ['Demand ($\\varepsilon^d$)',
                'Supply ($\\varepsilon^s$)',
                'Monetary ($\\varepsilon^m$)']

for idx, (ax, vname) in enumerate(zip(axes_fevd, var_names_short)):
    bottom = np.zeros(n_ahead + 1)
    for j in range(n_vars):
        ax.bar(h_range, fevd_array[:, idx, j], bottom=bottom,
               color=shock_colors[j], alpha=0.8, label=shock_labels[j],
               width=0.8)
        bottom += fevd_array[:, idx, j]
    ax.set_title(vname)
    ax.set_xlabel('Horizon (quarters)')
    ax.set_ylabel('Share of FEV')
    ax.set_ylim(0, 1.05)
    if idx == 0:
        ax.legend(fontsize=8, loc='upper right')

plt.tight_layout()
savefig('wrapup_FEVD_cholesky.pdf', fig_fevd)

# -- Print selected horizons -----------------------------------------
print("\nFEVD at selected horizons (fraction of variance explained):")
for i, vname in enumerate(var_names_short):
    print(f"\n  Variable: {vname}")
    print(f"    {'h':>3s}  {'demand':>8s}  {'supply':>8s}  {'monetary':>8s}")
    print("    " + "-" * 33)
    for h in [0, 1, 4, 8, 20]:
        if h <= n_ahead:
            print(f"    {h:3d}  {fevd_array[h, i, 0]:8.3f}  "
                  f"{fevd_array[h, i, 1]:8.3f}  {fevd_array[h, i, 2]:8.3f}")

print("""
Key lesson:
  * At h=0, the Cholesky ordering attributes 100% of interest-rate
    forecast error to all three shocks (g and pi contribute via the
    Taylor rule), while g is driven entirely by demand shocks.
  * At longer horizons, monetary shocks explain a non-trivial share
    of output gap variation (the transmission mechanism operates).
  * FEVD complements IRFs: IRFs show the DIRECTION of effects,
    FEVD shows the RELATIVE IMPORTANCE of each shock.
""")


# =====================================================================
# 7.  TRUE IRF (known from DGP) -- benchmark
# =====================================================================

print("=" * 70)
print("7. TRUE IRFs -- computed from structural parameters (benchmark)")
print("=" * 70)

# The true structural impact matrix (by construction of the DGP):
#   S = [[sigma_d,             0,            0       ],
#        [0,                   sigma_s,      0       ],
#        [phi_g*sigma_d,       phi_pi*sigma_s, sigma_m]]
#
# Because g_t depends on LAGGED r_{t-1} only, and pi_t depends on
# LAGGED g_{t-1} only, the contemporaneous structure is recursive.

S_true = np.array([
    [sigma_d,          0,         0      ],
    [0,                sigma_s,   0      ],
    [phi_g * sigma_d,  phi_pi * sigma_s,  sigma_m]
])

print("True structural impact matrix S (scaled by shock std devs):")
print(np.array2string(S_true, precision=4))

# True reduced-form companion matrix M:
# From the DGP:
#   g_t  =  rho_g * g_{t-1}  - alpha_r * (i_{t-1} - pi_{t-1})  + ...
#   pi_t =  kappa * g_{t-1}  + rho_pi * pi_{t-1}                + ...
#   i_t  =  phi_g * [rho_g * g_{t-1} - alpha_r * r_{t-1} + eps^d]
#          + phi_pi * [kappa * g_{t-1} + rho_pi * pi_{t-1} + eps^s]   + eps^m
#
# In matrix form: y_t = M * y_{t-1} + S * eps_t
# where y = (g, pi, i)

M_true = np.array([
    [rho_g,                              alpha_r,               -alpha_r         ],
    [kappa,                              rho_pi,                 0               ],
    [phi_g*rho_g + phi_pi*kappa,
     phi_g*alpha_r + phi_pi*rho_pi,
     -phi_g*alpha_r]
])

print("\nTrue reduced-form companion matrix M:")
print(np.array2string(M_true, precision=4))

# Verify stability: all eigenvalues of M should be inside the unit circle
eigenvals_true = np.abs(np.linalg.eigvals(M_true))
print(f"\nEigenvalue moduli of M: {np.round(eigenvals_true, 4)}")
print(f"DGP is stable: {np.all(eigenvals_true < 1)}")

# Compute true IRFs by iterating: Phi_h = M^h * S
true_irf = np.zeros((n_ahead + 1, 3, 3))
M_power = np.eye(3)
for h in range(n_ahead + 1):
    true_irf[h] = M_power @ S_true
    M_power = M_power @ M_true

# Normalise: response to a 1-unit monetary policy shock
# Column 3 of S_true has magnitude sigma_m in the (3,3) entry,
# so dividing by sigma_m gives the per-unit-shock response.
true_irf_g_m  = true_irf[:, 0, 2] / sigma_m
true_irf_pi_m = true_irf[:, 1, 2] / sigma_m
true_irf_i_m  = true_irf[:, 2, 2] / sigma_m

# Also normalise the Cholesky SVAR IRFs to per-unit-of-i_t shock
scale_chol = irf_cholesky_i[0] if abs(irf_cholesky_i[0]) > 1e-10 else 1.0
irf_cholesky_g_norm  = irf_cholesky_g  / scale_chol
irf_cholesky_pi_norm = irf_cholesky_pi / scale_chol
irf_cholesky_i_norm  = irf_cholesky_i  / scale_chol


# =====================================================================
# 8.  LP-IV & SVAR-IV -- External Instruments
# =====================================================================

print("\n" + "=" * 70)
print("8. EXTERNAL INSTRUMENTS -- LP-IV and SVAR-IV")
print("=" * 70)

# ── 8a. Align data for instrument regressions ───────────────────────
u_hat  = var_results.resid.values     # reduced-form residuals (T-p, 3)
n_obs  = len(u_hat)
p_lag  = p_opt
start  = p_lag                        # first usable observation index

z_iv   = z[start:][:n_obs]           # instrument aligned with residuals
i_iv   = var_data['interest_rate'].values[start:][:n_obs]

# Controls: lags of all variables (one lag for simplicity)
controls = var_data.shift(1).dropna().values
controls = controls[(start - 1):][:n_obs]

n_obs = min(len(z_iv), len(i_iv), len(controls))
z_iv      = z_iv[:n_obs]
i_iv      = i_iv[:n_obs]
controls  = controls[:n_obs]

# ── 8b. First-stage diagnostics ─────────────────────────────────────
X_fs = np.column_stack([z_iv, controls, np.ones(n_obs)])
gamma_hat = lstsq(X_fs, i_iv, rcond=None)[0]
i_hat     = X_fs @ gamma_hat
resid_fs  = i_iv - i_hat

# Restricted model (without the instrument)
X_fs_restricted = np.column_stack([controls, np.ones(n_obs)])
gamma_restricted = lstsq(X_fs_restricted, i_iv, rcond=None)[0]
SSR_restricted = np.sum((i_iv - X_fs_restricted @ gamma_restricted) ** 2)
SSR_full = np.sum(resid_fs ** 2)

F_stat = ((SSR_restricted - SSR_full) / 1) / (SSR_full / (n_obs - X_fs.shape[1]))
print(f"\nFirst-stage F-statistic: {F_stat:.1f}  (rule of thumb: F > 10)")
if F_stat > 10:
    print("  -> Strong instrument")
else:
    print("  -> WARNING: potentially weak instrument")

# Empirical first-stage R-squared (partial)
R2_partial = 1 - SSR_full / SSR_restricted
print(f"Partial R-squared: {R2_partial:.3f}  (theoretical: {R2_theory:.3f})")


# ── 8c. LP-IV with Newey-West standard errors ───────────────────────
print("\n-- 8c. LP-IV (Local Projections with IV) --")
print("    Using Newey-West (HAC) standard errors for valid inference.")

def newey_west_variance(X, residuals, n_lags):
    """
    Newey-West heteroskedasticity and autocorrelation consistent (HAC)
    variance estimator for OLS/2SLS.

    Parameters
    ----------
    X : array (n, k)   -- regressors (second-stage, using fitted values)
    residuals : array (n,) -- second-stage residuals
    n_lags : int        -- truncation lag for HAC

    Returns
    -------
    V_hat : array (k, k) -- HAC variance-covariance matrix of coefficients
    """
    n, k = X.shape
    # Meat: S_0 + sum of weighted cross-products
    # Start with the "HC0" part
    S = np.zeros((k, k))
    for j in range(n_lags + 1):
        w = 1.0 if j == 0 else 1.0 - j / (n_lags + 1)  # Bartlett kernel
        for t in range(j, n):
            x_t = X[t].reshape(-1, 1)
            x_s = X[t - j].reshape(-1, 1)
            contrib = residuals[t] * residuals[t - j] * (x_t @ x_s.T)
            S += w * contrib
            if j > 0:
                S += w * contrib.T  # symmetry
    # Bread: (X'X)^{-1}
    XtX_inv = np.linalg.inv(X.T @ X)
    V_hat = n / (n - k) * XtX_inv @ S @ XtX_inv
    return V_hat

outcomes_full = data[['output_gap', 'inflation', 'interest_rate']].values

lp_iv_g      = np.full(n_ahead + 1, np.nan)
lp_iv_pi     = np.full(n_ahead + 1, np.nan)
lp_iv_i      = np.full(n_ahead + 1, np.nan)
lp_iv_se_g   = np.full(n_ahead + 1, np.nan)
lp_iv_se_pi  = np.full(n_ahead + 1, np.nan)

for h in range(n_ahead + 1):
    end_idx = T - h
    n_h = min(n_obs, end_idx - start)
    if n_h <= 10:
        break

    y_g_h  = outcomes_full[start + h : start + h + n_h, 0]
    y_pi_h = outcomes_full[start + h : start + h + n_h, 1]
    y_i_h  = outcomes_full[start + h : start + h + n_h, 2]

    z_h = z_iv[:n_h]
    W_h = controls[:n_h]

    # First stage: i_t = gamma * z_t + delta' W_t + const + v_t
    X1 = np.column_stack([z_h, W_h, np.ones(n_h)])
    coef1     = lstsq(X1, i_iv[:n_h], rcond=None)[0]
    i_fitted  = X1 @ coef1

    # Second stage: Y_{t+h} = beta_h * i_hat_t + gamma' W_t + const + e_{t+h}
    X2 = np.column_stack([i_fitted, W_h, np.ones(n_h)])

    # Newey-West lag = floor(0.75 * h^{1/3}) + 1, at least 1
    nw_lag = max(int(0.75 * (h ** (1/3))), 1)

    for y_h, storage, se_storage in [
        (y_g_h,  lp_iv_g,  lp_iv_se_g),
        (y_pi_h, lp_iv_pi, lp_iv_se_pi),
        (y_i_h,  lp_iv_i,  None),
    ]:
        coef2 = lstsq(X2, y_h, rcond=None)[0]
        storage[h] = coef2[0]
        if se_storage is not None:
            resid_2s = y_h - X2 @ coef2
            try:
                V_nw = newey_west_variance(X2, resid_2s, nw_lag)
                se_storage[h] = np.sqrt(max(V_nw[0, 0], 0))
            except (np.linalg.LinAlgError, ValueError):
                se_storage[h] = np.nan

# Normalise LP-IV to per-unit-of-i_t shock
scale_lp = lp_iv_i[0] if abs(lp_iv_i[0]) > 1e-10 else 1.0
lp_iv_g_norm     = lp_iv_g  / scale_lp
lp_iv_pi_norm    = lp_iv_pi / scale_lp
lp_iv_se_g_norm  = lp_iv_se_g  / np.abs(scale_lp)
lp_iv_se_pi_norm = lp_iv_se_pi / np.abs(scale_lp)

print(f"\nLP-IV: monetary policy shock -> Output gap")
print(f"  Impact (h=0): {lp_iv_g_norm[0]:+.4f}")
print(f"  h=4:          {lp_iv_g_norm[min(4, n_ahead)]:+.4f}")


# ── 8d. SVAR-IV (Proxy VAR) ─────────────────────────────────────────
print("\n-- 8d. SVAR-IV (Proxy VAR) --")

# Step 1: reduced-form VAR residuals (already computed: u_hat)
u_i = u_hat[:, 2]   # residual of interest rate equation

# Step 2: first stage -- regress u_3 (interest rate residual) on z_t
slope_fs   = np.cov(z_iv[:len(u_i)], u_i)[0, 1] / np.var(z_iv[:len(u_i)])
u_i_fitted = slope_fs * z_iv[:len(u_i)]

# F-stat for SVAR-IV first stage
corr_zu = np.corrcoef(z_iv[:len(u_i)], u_i)[0, 1]
F_svariv = (len(u_i) - 2) * corr_zu**2 / (1 - corr_zu**2)
print(f"SVAR-IV first-stage F-statistic: {F_svariv:.1f}")

# Step 3: second stage -- regress u_1, u_2 on fitted u_3
# This identifies B_{*3} / B_{33} (relative impact effects)
b_g3  = np.cov(u_i_fitted, u_hat[:, 0])[0, 1] / np.var(u_i_fitted)
b_pi3 = np.cov(u_i_fitted, u_hat[:, 1])[0, 1] / np.var(u_i_fitted)

# Impact vector (normalised so that i responds by 1 on impact)
B_col3 = np.array([b_g3, b_pi3, 1.0])

print(f"\nSVAR-IV identified impact column (per unit i shock):")
print(f"  Output gap:      {B_col3[0]:+.4f}")
print(f"  Inflation:       {B_col3[1]:+.4f}")
print(f"  Interest rate:   {B_col3[2]:+.4f}  (normalised to 1)")

# True impact column for comparison
true_impact_norm = S_true[:, 2] / sigma_m
print(f"\nTrue impact column (per unit eps^m):")
print(f"  Output gap:      {true_impact_norm[0]:+.4f}")
print(f"  Inflation:       {true_impact_norm[1]:+.4f}")
print(f"  Interest rate:   {true_impact_norm[2]:+.4f}")

# Step 4: propagate through VAR dynamics
ma_coefs = var_results.ma_rep(maxn=n_ahead)  # shape (n_ahead+1, 3, 3)

svariv_irf = np.zeros((n_ahead + 1, 3))
for h in range(n_ahead + 1):
    svariv_irf[h] = ma_coefs[h] @ B_col3

svariv_irf_g  = svariv_irf[:, 0]
svariv_irf_pi = svariv_irf[:, 1]
svariv_irf_i  = svariv_irf[:, 2]

print(f"\nSVAR-IV: monetary policy shock -> Output gap")
print(f"  Impact (h=0): {svariv_irf_g[0]:+.4f}")
print(f"  h=4:          {svariv_irf_g[min(4, n_ahead)]:+.4f}")


# =====================================================================
# 9.  COMPARISON -- True IRF vs Cholesky vs LP-IV vs SVAR-IV
# =====================================================================

print("\n" + "=" * 70)
print("9. COMPARISON -- Four IRF estimates for the monetary policy shock")
print("=" * 70)

horizons = np.arange(n_ahead + 1)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Monetary Policy Shock: True IRF vs Estimated IRFs", fontsize=13)

var_names_plot = ['Output gap $g_t$', 'Inflation $\\pi_t$',
                  'Interest rate $i_t$']
true_irfs = [true_irf_g_m, true_irf_pi_m, true_irf_i_m]
chol_irfs = [irf_cholesky_g_norm, irf_cholesky_pi_norm, irf_cholesky_i_norm]
lpiv_irfs = [lp_iv_g_norm, lp_iv_pi_norm, lp_iv_i / scale_lp]
sviv_irfs = [svariv_irf_g, svariv_irf_pi, svariv_irf_i]
se_list   = [lp_iv_se_g_norm, lp_iv_se_pi_norm, None]
boot_ci_lo_list = [boot_ci_g_lo, boot_ci_pi_lo, None]
boot_ci_hi_list = [boot_ci_g_hi, boot_ci_pi_hi, None]

for ax, vname, true_ir, chol_ir, lp_ir, sv_ir, se_ir, blo, bhi in zip(
        axes, var_names_plot, true_irfs, chol_irfs, lpiv_irfs, sviv_irfs,
        se_list, boot_ci_lo_list, boot_ci_hi_list):
    ax.plot(horizons, true_ir, 'k-',  linewidth=2.5, label='True IRF')
    ax.plot(horizons, chol_ir, 'b--', linewidth=1.5, label='Cholesky SVAR')
    ax.plot(horizons, lp_ir,   'r:',  linewidth=1.5, label='LP-IV')
    ax.plot(horizons, sv_ir,   'g-.', linewidth=1.5, label='SVAR-IV')
    if blo is not None:
        ax.fill_between(horizons, blo, bhi,
                        alpha=0.12, color='blue', label='Cholesky 90% CI (boot)')
    if se_ir is not None:
        ax.fill_between(horizons,
                        lp_ir - 1.65 * se_ir,
                        lp_ir + 1.65 * se_ir,
                        alpha=0.15, color='red', label='LP-IV 90% CI (NW)')
    ax.axhline(0, color='grey', linewidth=0.5)
    ax.set_title(vname)
    ax.set_xlabel('Horizon (quarters)')
    if ax == axes[0]:
        ax.legend(fontsize=7, loc='best')

plt.tight_layout()
savefig('wrapup_IRF_comparison.pdf')

# -- Numerical comparison table --------------------------------------
print("\nResponse of OUTPUT GAP to a +1 unit monetary policy shock:")
print(f"{'h':>3s}  {'True':>8s}  {'Cholesky':>8s}  {'LP-IV':>8s}  {'SVAR-IV':>8s}")
print("-" * 45)
for h in [0, 1, 2, 4, 8, 12, 16, 20]:
    if h <= n_ahead:
        print(f"{h:3d}  {true_irf_g_m[h]:+8.4f}  "
              f"{irf_cholesky_g_norm[h]:+8.4f}  "
              f"{lp_iv_g_norm[h]:+8.4f}  {svariv_irf_g[h]:+8.4f}")

print("\nResponse of INFLATION to a +1 unit monetary policy shock:")
print(f"{'h':>3s}  {'True':>8s}  {'Cholesky':>8s}  {'LP-IV':>8s}  {'SVAR-IV':>8s}")
print("-" * 45)
for h in [0, 1, 2, 4, 8, 12, 16, 20]:
    if h <= n_ahead:
        print(f"{h:3d}  {true_irf_pi_m[h]:+8.4f}  "
              f"{irf_cholesky_pi_norm[h]:+8.4f}  "
              f"{lp_iv_pi_norm[h]:+8.4f}  {svariv_irf_pi[h]:+8.4f}")


# =====================================================================
# 10. WHAT GOES WRONG WITH THE WRONG CHOLESKY ORDERING
# =====================================================================

print("\n" + "=" * 70)
print("10. WRONG CHOLESKY ORDERING -- ordering (i, g, pi)")
print("=" * 70)
print("""
What happens if we order the interest rate FIRST?

This imposes that the interest rate is "predetermined" and that the
output gap can respond contemporaneously to monetary policy shocks.
But it also means the "monetary policy shock" is now identified as
the FIRST shock (interest-rate innovation orthogonalised first),
which contaminates it with systematic Taylor-rule responses.

This is a common mistake and a source of the "price puzzle":
a tightening shock appears to RAISE inflation because the identified
shock actually includes the central bank's endogenous response to
expected inflation.
""")

# Reorder: (i, g, pi) instead of (g, pi, i)
wrong_data = var_data[['interest_rate', 'output_gap', 'inflation']].copy()

A_wrong = np.array([
    [1,   0,   0],
    ['E', 1,   0],
    ['E', 'E', 1]
])

# Instead of using statsmodels' SVAR (which internally optimises and may
# converge to the same decomposition), we compute the Cholesky decomposition
# of the residual covariance matrix directly.  This is the most transparent
# way to show how ordering affects identification.

var_wrong_model   = VAR(wrong_data)
var_wrong_results = var_wrong_model.fit(maxlags=p_opt, trend='c')

# Residual covariance in the wrong ordering (i, g, pi)
Omega_wrong = np.array(var_wrong_results.sigma_u)
print("Residual covariance matrix (wrong ordering: i, g, pi):")
print(np.array2string(Omega_wrong, precision=6))

# Cholesky decomposition: Omega = P P'  (P lower-triangular)
P_wrong = np.linalg.cholesky(Omega_wrong)
print("\nCholesky factor P (wrong ordering):")
print(np.array2string(P_wrong, precision=6))

# For comparison, Cholesky in the correct ordering (g, pi, i)
Omega_correct = np.array(var_results.sigma_u)
P_correct = np.linalg.cholesky(Omega_correct)
print("\nCholesky factor P (correct ordering: g, pi, i):")
print(np.array2string(P_correct, precision=6))

# Compute IRFs for the wrong ordering using MA representation
ma_wrong = var_wrong_results.ma_rep(maxn=n_ahead)

# Structural IRFs = Psi_h * P   for each horizon h
# In the wrong ordering, shock 0 is the "monetary policy shock"
# (interest_rate innovation, not orthogonalised against g or pi)
wrong_struct_irf = np.zeros((n_ahead + 1, 3, 3))
for h in range(n_ahead + 1):
    wrong_struct_irf[h] = ma_wrong[h] @ P_wrong

# Extract monetary policy shock (shock 0 in wrong ordering)
# Variables in wrong ordering: 0=i, 1=g, 2=pi
wrong_irf_i  = wrong_struct_irf[:, 0, 0]  # i response to i shock
wrong_irf_g  = wrong_struct_irf[:, 1, 0]  # g response to i shock
wrong_irf_pi = wrong_struct_irf[:, 2, 0]  # pi response to i shock

# Normalise
scale_wrong = wrong_irf_i[0] if abs(wrong_irf_i[0]) > 1e-10 else 1.0
wrong_irf_g_norm  = wrong_irf_g  / scale_wrong
wrong_irf_pi_norm = wrong_irf_pi / scale_wrong
wrong_irf_i_norm  = wrong_irf_i  / scale_wrong

print(f"\nWrong ordering -- impact of 'monetary policy shock' (shock 0):")
print(f"  On interest rate (h=0): {wrong_irf_i[0]:+.4f}")
print(f"  On output gap    (h=0): {wrong_irf_g[0]:+.4f}")
print(f"  On inflation     (h=0): {wrong_irf_pi[0]:+.4f}")
print(f"\nAfter normalisation (per unit i shock):")
print(f"  On output gap    (h=0): {wrong_irf_g_norm[0]:+.4f}")
print(f"  On inflation     (h=0): {wrong_irf_pi_norm[0]:+.4f}")

# -- Figure: correct vs wrong ordering --------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Effect of Cholesky Ordering on Identified IRFs", fontsize=13)

correct_list = [irf_cholesky_g_norm, irf_cholesky_pi_norm, irf_cholesky_i_norm]
wrong_list   = [wrong_irf_g_norm, wrong_irf_pi_norm, wrong_irf_i_norm]

for ax, vname, true_ir, correct_ir, wrong_ir in zip(
        axes, var_names_plot, true_irfs, correct_list, wrong_list):
    ax.plot(horizons, true_ir,    'k-',  linewidth=2.5, label='True IRF')
    ax.plot(horizons, correct_ir, 'b--', linewidth=1.5,
            label='Correct: (g, pi, i)')
    ax.plot(horizons, wrong_ir,   'r-',  linewidth=1.5,
            label='Wrong: (i, g, pi)')
    ax.axhline(0, color='grey', linewidth=0.5)
    ax.set_title(vname)
    ax.set_xlabel('Horizon (quarters)')
    if ax == axes[0]:
        ax.legend(fontsize=8, loc='best')

plt.tight_layout()
savefig('wrapup_wrong_ordering.pdf')

# -- Numerical comparison: correct vs wrong ---------------------------
print("\nResponse of OUTPUT GAP: correct vs wrong ordering")
print(f"{'h':>3s}  {'True':>8s}  {'Correct':>8s}  {'Wrong':>8s}")
print("-" * 33)
for h in [0, 1, 2, 4, 8, 12]:
    if h <= n_ahead:
        print(f"{h:3d}  {true_irf_g_m[h]:+8.4f}  "
              f"{irf_cholesky_g_norm[h]:+8.4f}  "
              f"{wrong_irf_g_norm[h]:+8.4f}")

print("\nResponse of INFLATION: correct vs wrong ordering")
print(f"{'h':>3s}  {'True':>8s}  {'Correct':>8s}  {'Wrong':>8s}")
print("-" * 33)
for h in [0, 1, 2, 4, 8, 12]:
    if h <= n_ahead:
        print(f"{h:3d}  {true_irf_pi_m[h]:+8.4f}  "
              f"{irf_cholesky_pi_norm[h]:+8.4f}  "
              f"{wrong_irf_pi_norm[h]:+8.4f}")

print("""
Key lesson:
  * The WRONG ordering produces a biased "monetary policy shock":
    it mixes the true policy shock with endogenous Taylor-rule responses.
  * With real data, we do not know the true ordering, which is why
    external instruments (LP-IV, SVAR-IV) are so valuable -- they do
    not require a correct Cholesky ordering.
  * This is the fundamental motivation behind the proxy-VAR literature
    (Stock & Watson 2018, Gertler & Karadi 2015).
""")


# =====================================================================
# 11. NON-RECURSIVE DGP -- When Cholesky fails but IV succeeds
# =====================================================================

print("\n" + "=" * 70)
print("11. NON-RECURSIVE DGP -- Cholesky fails, IV succeeds")
print("=" * 70)
print("""
We now modify the DGP so that inflation responds to the CONTEMPORANEOUS
interest rate (e.g. an exchange-rate pass-through or forward-looking
channel). This breaks the recursive structure: even the "correct"
ordering (g, pi, i) no longer identifies the monetary policy shock,
because pi_t now depends on i_t within the period.

Modified Phillips curve:
  pi_t = rho_pi * pi_{t-1} + kappa * g_{t-1} + delta_i * i_t + eps^s_t

where delta_i > 0 captures the contemporaneous pass-through.
""")

delta_i = 0.20   # contemporaneous inflation response to interest rate

# -- Simulate the non-recursive economy --------------------------------
g_nr      = np.zeros(T)
pi_nr     = np.zeros(T)
i_nr      = np.zeros(T)
r_nr      = np.zeros(T)

for t in range(1, T):
    g_nr[t]  = rho_g * g_nr[t-1] - alpha_r * r_nr[t-1] + eps_d[t]
    # Taylor rule (same shocks for comparability)
    i_nr[t]  = phi_pi * (rho_pi * pi_nr[t-1] + kappa * g_nr[t-1] + eps_s[t]) \
               / (1 - phi_pi * delta_i) \
               + phi_g * g_nr[t] / (1 - phi_pi * delta_i) \
               + eps_m[t] / (1 - phi_pi * delta_i)
    # Non-recursive Phillips curve: pi depends on contemporaneous i
    pi_nr[t] = rho_pi * pi_nr[t-1] + kappa * g_nr[t-1] \
               + delta_i * i_nr[t] + eps_s[t]
    r_nr[t]  = i_nr[t] - pi_nr[t]

nr_data = pd.DataFrame({
    'output_gap':    g_nr,
    'inflation':     pi_nr,
    'interest_rate': i_nr,
}, index=data.index)

# -- Cholesky SVAR on non-recursive data --------------------------------
nr_var = VAR(nr_data)
nr_var_res = nr_var.fit(p_opt, trend='c')
Omega_nr = np.array(nr_var_res.sigma_u)
P_nr = np.linalg.cholesky(Omega_nr)
ma_nr = nr_var_res.ma_rep(maxn=n_ahead)

# Cholesky IRFs: monetary shock = column 2 (same ordering: g, pi, i)
nr_chol_irf = np.zeros((n_ahead + 1, 3))
for h in range(n_ahead + 1):
    nr_chol_irf[h] = (ma_nr[h] @ P_nr)[:, 2]
scale_nr_chol = nr_chol_irf[0, 2] if abs(nr_chol_irf[0, 2]) > 1e-10 else 1.0
nr_chol_g_norm  = nr_chol_irf[:, 0] / scale_nr_chol
nr_chol_pi_norm = nr_chol_irf[:, 1] / scale_nr_chol

# -- True IRF for non-recursive DGP ------------------------------------
# Contemporaneous impact matrix S_nr (NOT lower-triangular):
#   g:  only eps_d on impact     -> row = [sigma_d, 0, 0]
#   i:  depends on g_t and eps_s through pi -> complex
#   pi: depends on i_t -> depends on eps_m
# We solve the simultaneous system:
#   pi = ... + delta_i * i + eps_s
#   i  = phi_pi * pi + phi_g * g + eps_m
# Substituting: pi = ... + delta_i*(phi_pi*pi + phi_g*g + eps_m) + eps_s
#   pi*(1 - delta_i*phi_pi) = ... + delta_i*phi_g*g + delta_i*eps_m + eps_s
#   pi = ... + [delta_i*phi_g/(1-delta_i*phi_pi)]*sigma_d*e_d
#        + [1/(1-delta_i*phi_pi)]*sigma_s*e_s
#        + [delta_i/(1-delta_i*phi_pi)]*sigma_m*e_m
denom_nr = 1.0 - delta_i * phi_pi
S_nr_true = np.array([
    [sigma_d,   0,                              0],
    [delta_i * phi_g * sigma_d / denom_nr,
     sigma_s / denom_nr,
     delta_i * sigma_m / denom_nr],
    [phi_g * sigma_d / denom_nr + phi_pi * delta_i * phi_g * sigma_d / denom_nr,
     phi_pi * sigma_s / denom_nr,
     sigma_m / denom_nr],
])
# Simplify row 2 (i): phi_g*sigma_d*(1+phi_pi*delta_i)/denom = phi_g*sigma_d/denom_nr
# Actually let's recompute carefully:
# i = phi_pi*pi + phi_g*g + eps_m
# i = phi_pi*[...] + phi_g*sigma_d*e_d + eps_m
# where pi's impact on e_d is delta_i*phi_g*sigma_d/denom_nr
# so i's impact on e_d = phi_pi*(delta_i*phi_g*sigma_d/denom_nr) + phi_g*sigma_d
#                       = phi_g*sigma_d*(phi_pi*delta_i/denom_nr + 1)
#                       = phi_g*sigma_d*(phi_pi*delta_i + denom_nr)/denom_nr
#                       = phi_g*sigma_d / denom_nr
S_nr_true = np.array([
    [sigma_d,                              0,                          0],
    [delta_i * phi_g * sigma_d / denom_nr, sigma_s / denom_nr,         delta_i * sigma_m / denom_nr],
    [phi_g * sigma_d / denom_nr,           phi_pi * sigma_s / denom_nr, sigma_m / denom_nr],
])

print("True impact matrix S (non-recursive DGP):")
print(np.array2string(S_nr_true, precision=4))
print("  Note: S is NOT lower-triangular -> Cholesky is misspecified.")

# True companion matrix for non-recursive DGP
M_nr_true = np.array([
    [rho_g,                          alpha_r,                    -alpha_r],
    [kappa + delta_i*phi_g*rho_g/denom_nr,
     rho_pi/denom_nr + delta_i*(phi_g*alpha_r + phi_pi*rho_pi)/denom_nr,
     -delta_i*phi_g*alpha_r/denom_nr],
    [phi_g*rho_g/denom_nr + phi_pi*kappa/denom_nr,
     (phi_g*alpha_r + phi_pi*rho_pi)/denom_nr,
     -phi_g*alpha_r/denom_nr],
])

# True IRFs for non-recursive DGP
nr_true_irf = np.zeros((n_ahead + 1, 3, 3))
M_nr_power = np.eye(3)
for h in range(n_ahead + 1):
    nr_true_irf[h] = M_nr_power @ S_nr_true
    M_nr_power = M_nr_power @ M_nr_true

nr_true_g_m  = nr_true_irf[:, 0, 2] / (sigma_m / denom_nr)
nr_true_pi_m = nr_true_irf[:, 1, 2] / (sigma_m / denom_nr)

# -- SVAR-IV on non-recursive data (should still work) ------------------
nr_resid = nr_var_res.resid.values
n_nr = len(nr_resid)
z_nr = z[p_opt:][:n_nr]
u_i_nr = nr_resid[:, 2]

slope_nr   = np.cov(z_nr, u_i_nr)[0, 1] / np.var(z_nr)
u_i_nr_fit = slope_nr * z_nr

b_g3_nr  = np.cov(u_i_nr_fit, nr_resid[:, 0])[0, 1] / np.var(u_i_nr_fit)
b_pi3_nr = np.cov(u_i_nr_fit, nr_resid[:, 1])[0, 1] / np.var(u_i_nr_fit)
B_nr = np.array([b_g3_nr, b_pi3_nr, 1.0])

ma_nr_rep = nr_var_res.ma_rep(maxn=n_ahead)
nr_svariv_irf = np.zeros((n_ahead + 1, 3))
for h in range(n_ahead + 1):
    nr_svariv_irf[h] = ma_nr_rep[h] @ B_nr

nr_svariv_g  = nr_svariv_irf[:, 0]
nr_svariv_pi = nr_svariv_irf[:, 1]

# -- Figure: non-recursive DGP comparison ------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Non-Recursive DGP: Cholesky Fails, SVAR-IV Succeeds", fontsize=13)

for ax, vname, true_ir, chol_ir, sviv_ir in zip(
        axes,
        ['Output gap $g_t$', 'Inflation $\\pi_t$'],
        [nr_true_g_m, nr_true_pi_m],
        [nr_chol_g_norm, nr_chol_pi_norm],
        [nr_svariv_g, nr_svariv_pi]):
    ax.plot(horizons, true_ir, 'k-',  linewidth=2.5, label='True IRF')
    ax.plot(horizons, chol_ir, 'b--', linewidth=1.5, label='Cholesky (g,pi,i)')
    ax.plot(horizons, sviv_ir, 'g-.', linewidth=1.5, label='SVAR-IV')
    ax.axhline(0, color='grey', linewidth=0.5)
    ax.set_title(vname)
    ax.set_xlabel('Horizon (quarters)')
    if ax == axes[0]:
        ax.legend(fontsize=8, loc='best')

plt.tight_layout()
savefig('wrapup_nonrecursive_DGP.pdf')

print("""
Key lesson:
  * When the DGP is NOT recursive, Cholesky identification fails even
    with the "correct" ordering (g, pi, i), because the true impact
    matrix is not lower-triangular.
  * SVAR-IV still recovers the correct IRF, because it relies on the
    external instrument (relevance + exogeneity), NOT on the ordering.
  * This is the fundamental advantage of external-instrument methods:
    they do not require knowledge of the contemporaneous causal structure.
""")


# =====================================================================
# 12. CONTAMINATED INSTRUMENT -- What happens when exogeneity fails
# =====================================================================

print("=" * 70)
print("12. CONTAMINATED INSTRUMENT -- violation of exogeneity")
print("=" * 70)
print("""
We now construct a "contaminated" instrument that violates exogeneity:

  z^c_t = eps^m_t + eta_t + lambda * eps^d_t

The parameter lambda > 0 introduces correlation between the instrument
and the demand shock. This mimics the "information effect" concern
(Nakamura & Steinsson 2018): if the high-frequency surprise window
also captures macro news (demand shocks), the instrument is invalid.
""")

lambda_contam = 0.40   # contamination strength

# Contaminated instrument
z_contam = eps_m + eta + lambda_contam * eps_d

# -- LP-IV with contaminated instrument --------------------------------
z_contam_iv = z_contam[start:][:n_obs]

contam_lp_g = np.full(n_ahead + 1, np.nan)
contam_lp_i = np.full(n_ahead + 1, np.nan)

for h in range(n_ahead + 1):
    end_idx = T - h
    n_h = min(n_obs, end_idx - start)
    if n_h <= 10:
        break
    y_g_h = outcomes_full[start + h : start + h + n_h, 0]
    y_i_h = outcomes_full[start + h : start + h + n_h, 2]
    z_h = z_contam_iv[:n_h]
    W_h = controls[:n_h]
    X1 = np.column_stack([z_h, W_h, np.ones(n_h)])
    coef1 = lstsq(X1, i_iv[:n_h], rcond=None)[0]
    i_fitted = X1 @ coef1
    X2 = np.column_stack([i_fitted, W_h, np.ones(n_h)])
    coef_g = lstsq(X2, y_g_h, rcond=None)[0]
    coef_i = lstsq(X2, y_i_h, rcond=None)[0]
    contam_lp_g[h] = coef_g[0]
    contam_lp_i[h] = coef_i[0]

scale_contam = contam_lp_i[0] if abs(contam_lp_i[0]) > 1e-10 else 1.0
contam_lp_g_norm = contam_lp_g / scale_contam

# -- SVAR-IV with contaminated instrument --------------------------------
z_contam_svariv = z_contam[start:][:len(u_hat)]
slope_contam = np.cov(z_contam_svariv, u_i)[0, 1] / np.var(z_contam_svariv)
u_i_fit_contam = slope_contam * z_contam_svariv
b_g3_contam  = np.cov(u_i_fit_contam, u_hat[:, 0])[0, 1] / np.var(u_i_fit_contam)
b_pi3_contam = np.cov(u_i_fit_contam, u_hat[:, 1])[0, 1] / np.var(u_i_fit_contam)
B_contam = np.array([b_g3_contam, b_pi3_contam, 1.0])

contam_svariv_irf = np.zeros((n_ahead + 1, 3))
for h in range(n_ahead + 1):
    contam_svariv_irf[h] = ma_coefs[h] @ B_contam
contam_svariv_g = contam_svariv_irf[:, 0]

# -- Figure: contaminated vs valid instrument ---------------------------
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
fig.suptitle(f"Effect of Instrument Contamination ($\\lambda$={lambda_contam})",
             fontsize=13)
ax.plot(horizons, true_irf_g_m, 'k-', linewidth=2.5, label='True IRF')
ax.plot(horizons, lp_iv_g_norm, 'g--', linewidth=1.5, label='LP-IV (valid z)')
ax.plot(horizons, contam_lp_g_norm, 'r-', linewidth=1.5,
        label=f'LP-IV (contaminated, $\\lambda$={lambda_contam})')
ax.plot(horizons, contam_svariv_g, 'm:', linewidth=1.5,
        label=f'SVAR-IV (contaminated)')
ax.axhline(0, color='grey', linewidth=0.5)
ax.set_title('Output gap response to monetary policy shock')
ax.set_xlabel('Horizon (quarters)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
savefig('wrapup_contaminated_instrument.pdf')

# -- Numerical comparison -----------------------------------------------
print("\nResponse of OUTPUT GAP: valid vs contaminated instrument")
print(f"{'h':>3s}  {'True':>8s}  {'LP-IV':>10s}  {'LP-IV(c)':>10s}  {'SVAR-IV(c)':>10s}")
print("-" * 50)
for h in [0, 1, 2, 4, 8, 12]:
    if h <= n_ahead:
        print(f"{h:3d}  {true_irf_g_m[h]:+8.4f}  "
              f"{lp_iv_g_norm[h]:+10.4f}  "
              f"{contam_lp_g_norm[h]:+10.4f}  "
              f"{contam_svariv_g[h]:+10.4f}")

print(f"""
Key lesson:
  * When the instrument correlates with demand shocks (lambda={lambda_contam}),
    both LP-IV and SVAR-IV produce BIASED estimates.
  * The bias is towards LESS contractionary effects: the contaminated
    instrument partly captures expansionary demand shocks alongside
    the contractionary monetary shock, attenuating the estimated effect.
  * Instrument validity (exogeneity) is critical and UNTESTABLE in
    practice -- it must be justified by economic reasoning about the
    identification strategy (e.g., the width of the surprise window).
  * This is the "information effect" concern of Nakamura & Steinsson
    (2018): high-frequency surprises may capture central bank information
    about the state of the economy, violating exogeneity.
""")


# =====================================================================
# 13. MONTE CARLO REPETITIONS -- Sampling distributions
# =====================================================================

print("=" * 70)
print("13. MONTE CARLO -- Sampling distributions of IRF estimates")
print("=" * 70)

n_mc = 500
h_eval = 4  # horizon at which we evaluate the IRF
mc_results = {
    'cholesky_correct': np.full(n_mc, np.nan),
    'cholesky_wrong':   np.full(n_mc, np.nan),
    'lpiv':             np.full(n_mc, np.nan),
    'svariv':           np.full(n_mc, np.nan),
}

print(f"\nRunning {n_mc} Monte Carlo replications (this may take a moment)...")

for mc in range(n_mc):
    # Draw new structural shocks
    e_d_mc = np.random.normal(0, sigma_d, T)
    e_s_mc = np.random.normal(0, sigma_s, T)
    e_m_mc = np.random.normal(0, sigma_m, T)
    eta_mc = np.random.normal(0, sigma_eta, T)

    # Simulate economy (recursive DGP)
    g_mc = np.zeros(T)
    pi_mc = np.zeros(T)
    i_mc = np.zeros(T)
    r_mc = np.zeros(T)
    for t in range(1, T):
        g_mc[t]  = rho_g * g_mc[t-1] - alpha_r * r_mc[t-1] + e_d_mc[t]
        pi_mc[t] = rho_pi * pi_mc[t-1] + kappa * g_mc[t-1] + e_s_mc[t]
        i_mc[t]  = phi_pi * pi_mc[t] + phi_g * g_mc[t] + e_m_mc[t]
        r_mc[t]  = i_mc[t] - pi_mc[t]

    z_mc = e_m_mc + eta_mc
    mc_df = pd.DataFrame({
        'output_gap': g_mc, 'inflation': pi_mc, 'interest_rate': i_mc
    })

    try:
        # --- Cholesky SVAR (correct ordering) ---
        var_mc = VAR(mc_df)
        res_mc = var_mc.fit(p_opt, trend='c')
        Omega_mc = np.array(res_mc.sigma_u)
        P_mc = np.linalg.cholesky(Omega_mc)
        ma_mc = res_mc.ma_rep(maxn=h_eval)
        chol_irf_h = (ma_mc[h_eval] @ P_mc)[:, 2]
        scale_mc = (np.eye(3) @ P_mc)[2, 2]
        mc_results['cholesky_correct'][mc] = chol_irf_h[0] / scale_mc

        # --- Cholesky SVAR (wrong ordering: i, g, pi) ---
        wrong_df = mc_df[['interest_rate', 'output_gap', 'inflation']]
        var_w_mc = VAR(wrong_df)
        res_w_mc = var_w_mc.fit(p_opt, trend='c')
        Omega_w = np.array(res_w_mc.sigma_u)
        P_w = np.linalg.cholesky(Omega_w)
        ma_w = res_w_mc.ma_rep(maxn=h_eval)
        wrong_irf_h = (ma_w[h_eval] @ P_w)[:, 0]  # shock 0 in wrong order
        scale_w = P_w[0, 0]
        mc_results['cholesky_wrong'][mc] = wrong_irf_h[1] / scale_w  # g response

        # --- SVAR-IV ---
        u_mc = res_mc.resid.values
        n_mc_obs = len(u_mc)
        z_mc_iv = z_mc[p_opt:][:n_mc_obs]
        u_i_mc = u_mc[:, 2]
        sl_mc = np.cov(z_mc_iv, u_i_mc)[0, 1] / np.var(z_mc_iv)
        u_fit_mc = sl_mc * z_mc_iv
        b_g_mc = np.cov(u_fit_mc, u_mc[:, 0])[0, 1] / np.var(u_fit_mc)
        B_mc = np.array([b_g_mc, 0.0, 1.0])  # only need g response
        mc_results['svariv'][mc] = (ma_mc[h_eval] @ B_mc)[0]

        # --- LP-IV ---
        ctrl_mc = mc_df.shift(1).dropna().values[(p_opt-1):][:n_mc_obs]
        n_lp = min(n_mc_obs, T - h_eval - p_opt)
        if n_lp > 10:
            outcomes_mc = mc_df.values
            y_g_h_mc = outcomes_mc[p_opt + h_eval : p_opt + h_eval + n_lp, 0]
            z_lp = z_mc_iv[:n_lp]
            i_lp = mc_df['interest_rate'].values[p_opt:][:n_lp]
            W_lp = ctrl_mc[:n_lp]
            X1_lp = np.column_stack([z_lp, W_lp, np.ones(n_lp)])
            c1_lp = lstsq(X1_lp, i_lp, rcond=None)[0]
            i_fit_lp = X1_lp @ c1_lp
            X2_lp = np.column_stack([i_fit_lp, W_lp, np.ones(n_lp)])
            c2_lp = lstsq(X2_lp, y_g_h_mc, rcond=None)[0]
            # Normalise by impact LP-IV for i
            y_i_0 = outcomes_mc[p_opt : p_opt + n_lp, 2]
            X1_0 = np.column_stack([z_mc_iv[:n_lp], ctrl_mc[:n_lp], np.ones(n_lp)])
            c1_0 = lstsq(X1_0, i_lp, rcond=None)[0]
            i_fit_0 = X1_0 @ c1_0
            X2_0 = np.column_stack([i_fit_0, ctrl_mc[:n_lp], np.ones(n_lp)])
            c_i_0 = lstsq(X2_0, y_i_0, rcond=None)[0]
            sc_lp = c_i_0[0] if abs(c_i_0[0]) > 1e-10 else 1.0
            mc_results['lpiv'][mc] = c2_lp[0] / sc_lp
    except Exception:
        pass

# True IRF at h_eval for output gap (per unit shock)
true_at_h = true_irf_g_m[h_eval]

# -- Figure: Monte Carlo distributions ----------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(f"Monte Carlo Distributions of Output Gap IRF at h={h_eval} "
             f"({n_mc} replications)", fontsize=13)

labels = ['Cholesky (correct)', 'Cholesky (wrong)', 'LP-IV', 'SVAR-IV']
keys = ['cholesky_correct', 'cholesky_wrong', 'lpiv', 'svariv']
colors = ['blue', 'red', 'orange', 'green']

for ax, key, label, color in zip(axes.flat, keys, labels, colors):
    vals = mc_results[key]
    vals = vals[~np.isnan(vals)]
    ax.hist(vals, bins=40, density=True, alpha=0.6, color=color, edgecolor='white')
    ax.axvline(true_at_h, color='black', linewidth=2, linestyle='-',
               label=f'True IRF = {true_at_h:.4f}')
    ax.axvline(np.median(vals), color=color, linewidth=1.5, linestyle='--',
               label=f'Median = {np.median(vals):.4f}')
    ax.set_title(label)
    ax.legend(fontsize=7)
    ax.set_xlabel(f'IRF(g, h={h_eval})')

plt.tight_layout()
savefig('wrapup_monte_carlo.pdf')

# -- Summary statistics ---------------------------------------------------
print(f"\nMonte Carlo results: output gap IRF at h={h_eval}")
print(f"  True value: {true_at_h:+.4f}")
print(f"{'Method':>22s}  {'Mean':>8s}  {'Median':>8s}  {'Std':>8s}  {'Bias':>8s}")
print("-" * 60)
for key, label in zip(keys, labels):
    vals = mc_results[key][~np.isnan(mc_results[key])]
    mn, md, sd = np.mean(vals), np.median(vals), np.std(vals)
    bias = mn - true_at_h
    print(f"{label:>22s}  {mn:+8.4f}  {md:+8.4f}  {sd:8.4f}  {bias:+8.4f}")

print("""
Key lessons from the Monte Carlo:
  * Cholesky (correct ordering) is UNBIASED with small variance --
    it works well when the DGP is truly recursive.
  * Cholesky (wrong ordering) is BIASED -- the mean is systematically
    shifted away from the true IRF, confirming that ordering errors
    produce inconsistent estimates.
  * LP-IV is UNBIASED but has LARGER VARIANCE than Cholesky or SVAR-IV,
    reflecting the efficiency cost of model-free estimation.
  * SVAR-IV is UNBIASED with SMALLER VARIANCE than LP-IV -- it gains
    efficiency by using the VAR dynamics, at the cost of model dependence.
  * This is the classic bias-variance trade-off in econometric
    identification.
""")


# =====================================================================
# 14. ECONOMIC INTERPRETATION AND KEY TAKEAWAYS
# =====================================================================

print("=" * 70)
print("14. ECONOMIC INTERPRETATION AND KEY TAKEAWAYS")
print("=" * 70)

print("""
This exercise simulates a textbook New-Keynesian economy and compares
four strategies for estimating the effect of monetary policy shocks:

+--------------------+------------------------------------------------+
| Method             | What it does                                   |
+--------------------+------------------------------------------------+
| True IRF           | Analytical benchmark from the known DGP.       |
+--------------------+------------------------------------------------+
| Cholesky SVAR      | Orders variables (g, pi, i) and uses the       |
|                    | triangular structure to identify shocks.        |
|                    | Works well HERE because the DGP is recursive.  |
+--------------------+------------------------------------------------+
| LP-IV              | Instruments the interest rate with a noisy      |
|                    | proxy z_t. Runs a separate 2SLS per horizon.    |
|                    | Model-free but noisier at long horizons.        |
+--------------------+------------------------------------------------+
| SVAR-IV            | Uses z_t to identify the impact column of the   |
| (Proxy VAR)        | VAR, then propagates via VAR dynamics.          |
|                    | More efficient than LP-IV but model-dependent.  |
+--------------------+------------------------------------------------+

Key results:
  1. All three methods should be close to the TRUE IRF, because:
     * The DGP is recursive -> Cholesky is correctly specified.
     * The instrument z_t is valid -> LP-IV and SVAR-IV are consistent.
  2. The WRONG Cholesky ordering (i first) produces biased IRFs.
     This illustrates the price-puzzle problem and why ordering matters.
  3. External instruments (LP-IV, SVAR-IV) remain valid regardless
     of the ordering, as long as the instrument is relevant and exogenous.
  4. When the DGP is NON-RECURSIVE (inflation responds to the
     contemporaneous interest rate), even the "correct" Cholesky
     ordering fails, while SVAR-IV remains consistent.
  5. When the instrument is CONTAMINATED (correlated with demand
     shocks), both LP-IV and SVAR-IV become biased -- instrument
     validity is critical and must be justified by economic reasoning.
  6. Monte Carlo simulations confirm: Cholesky (correct) and SVAR-IV
     are unbiased with small variance; LP-IV is unbiased but noisier;
     Cholesky (wrong) is systematically biased.
  7. FEVD reveals the relative importance of each shock in driving
     the variability of output, inflation, and the interest rate.

References:
  Christiano, Eichenbaum & Evans (1999, 2005) -- recursive SVAR
  Kuttner (2001) -- high-frequency monetary surprises
  Romer & Romer (2004) -- narrative monetary shocks
  Stock & Watson (2018) -- external instruments econometrics
  Gertler & Karadi (2015) -- proxy VAR application
  Jorda (2005) -- local projections
  Plagborg-Moller & Wolf (2021) -- LP and VAR estimate the same IRF
  Nakamura & Steinsson (2018) -- information effects
""")

print("=" * 70)
print("Figures saved to:", FIG_DIR)
print("=" * 70)
