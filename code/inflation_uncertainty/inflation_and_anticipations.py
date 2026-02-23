#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QMF — Inflation, Expectations, and Uncertainty
=============================================

This script accompanies the QMF lecture section "Inflation, Expectations, and Uncertainty".
It provides an end-to-end, reproducible workflow that:

1) Downloads two monthly French time series from Banque de France Webstat
   (Opendatasoft / Explore API v2.1) using an API key stored locally:

   - PAI press-based inflation expectations:
       series_key = "PAI.M.FR.N.PR._Z.INPR03.TX"
       Webstat catalogue (FR):
       https://webstat.banque-france.fr/fr/catalogue/pai/PAI.M.FR.N.PR._Z.INPR03.TX

   - HICP inflation (YoY):
       series_key = "ICP.M.FR.N.000000.4.ANR"
       Webstat catalogue (FR):
       https://webstat.banque-france.fr/fr/catalogue/icp/ICP.M.FR.N.000000.4.ANR

2) Cleans, aligns, and plots inflation and expectations on a common monthly timeline.

3) Tests key expectation-formation benchmarks:
   - Mincer–Zarnowitz regression: π_t = α + β π^e_t + ε_t
     * underreaction: H1: β < 1
     * intercept bias: H1: α ≠ 0
     * joint unbiasedness: H0: (α=0, β=1), Wald test
     Inference uses HAC (Newey–West) with 12 lags (monthly frequency).

   - Cointegration diagnostics (Engle–Granger; Johansen trace test) to assess whether
     inflation and expectations share a long-run relation (relevant if both are I(1)).

   - Coibion–Gorodnichenko information rigidity test:
       FE_t = π_t − π^e_t,  FE_t = c + λ FE_{t−1} + ε_t
     with a one-sided rigidity hypothesis H1: λ > 0 (HAC SE, 12 lags).

4) Links uncertainty to expectations:
   - Estimates inflation uncertainty as conditional volatility from a GARCH(1,1) model.
   - Regresses expectations on inflation and volatility (static and dynamic variants),
     including joint Wald tests on volatility terms.
   - Tests whether volatility amplifies rigidity via an interaction in the FE regression.
   - Provides system evidence via a VAR and Granger causality test (σ → π^e).

Data access & reproducibility
-----------------------------
- Requires a valid Webstat API key in a local text file (default: "APIwebstat.txt"),
  read from the working directory set near the top of the script.
- The script is designed for pedagogical clarity; empirical extensions may include:
  lag selection via IC, alternative volatility models (EGARCH/GJR), and additional controls.

Author: Eric Vansteenberghe
Course: Quantitative Methods in Finance (QMF)
License: MIT
"""

from pathlib import Path
from scipy import stats
import numpy as np
import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import statsmodels.api as sm
from statsmodels.stats.contrast import WaldTestResults
from statsmodels.tsa.vector_ar.vecm import VECM
from arch import arch_model
from statsmodels.tsa.api import VAR


# ------------------------------------------------------------
# 0) Working directory (so APIwebstat.txt is found)
# ------------------------------------------------------------
os.chdir('/Users/skimeur/Mon Drive/QMF/')

def load_webstat_key(path="APIwebstat.txt") -> str:
    key_path = Path(path).resolve()
    if not key_path.exists():
        raise FileNotFoundError(f"API key file not found at {key_path}")
    return key_path.read_text().strip()

# ------------------------------------------------------------
# 1) Fetch JSON export for the PAI series
# ------------------------------------------------------------
APIKEY = load_webstat_key("APIwebstat.txt")

url = "https://webstat.banque-france.fr/api/explore/v2.1/catalog/datasets/observations/exports/json/"
headers = {"Authorization": f"Apikey {APIKEY}"}

params = {
    "where": 'series_key IN ("PAI.M.FR.N.PR._Z.INPR03.TX")',
    "order_by": "-time_period_start",  # descending
}

r = requests.get(url, params=params, headers=headers, timeout=60)
r.raise_for_status()
data = r.json()

# ------------------------------------------------------------
# 2) JSON -> DataFrame (robust to list/dict payloads)
# ------------------------------------------------------------
def payload_to_rows(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "results" in payload and isinstance(payload["results"], list):
        return payload["results"]
    if isinstance(payload, dict):
        for v in payload.values():
            if isinstance(v, list):
                return v
    raise ValueError(f"Unexpected JSON payload shape: {type(payload)}")

rows = payload_to_rows(data)
df_raw = pd.DataFrame(rows)

# ------------------------------------------------------------
# 3) Keep relevant columns and clean types (PAI)
# ------------------------------------------------------------
required = {"time_period", "obs_value", "series_key", "title_fr"}
missing = required - set(df_raw.columns)
if missing:
    raise KeyError(f"Missing expected columns {sorted(missing)}. Got: {df_raw.columns.tolist()}")

VALUE_COL = "pai_exp_infl"   # short name for PAI obs_value

df_pai = df_raw[["time_period", "obs_value", "series_key", "title_fr"]].copy()
df_pai["date"] = pd.to_datetime(df_pai["time_period"], format="%Y-%m", errors="coerce")
df_pai[VALUE_COL] = pd.to_numeric(df_pai["obs_value"], errors="coerce")

title_pai = df_pai["title_fr"].dropna().iloc[0] if df_pai["title_fr"].notna().any() else "PAI series"

df_pai = (
    df_pai.dropna(subset=["date"])
          .sort_values("date")
          .set_index("date")
          [[VALUE_COL]]
)

print(df_pai.tail(10))

# ------------------------------------------------------------
# 3bis) Fetch + clean Inflation (ICP) series
# ------------------------------------------------------------
params_icp = {
    "where": 'series_key IN ("ICP.M.FR.N.000000.4.ANR")',
    "order_by": "-time_period_start",  # descending
}

r2 = requests.get(url, params=params_icp, headers=headers, timeout=60)
r2.raise_for_status()
data2 = r2.json()

rows2 = payload_to_rows(data2)
df_raw2 = pd.DataFrame(rows2)

missing2 = required - set(df_raw2.columns)
if missing2:
    raise KeyError(f"Missing expected columns {sorted(missing2)} in ICP. Got: {df_raw2.columns.tolist()}")

VALUE_COL_ICP = "inflation_yoy"  # short name for inflation obs_value

df_icp = df_raw2[["time_period", "obs_value", "series_key", "title_fr"]].copy()
df_icp["date"] = pd.to_datetime(df_icp["time_period"], format="%Y-%m", errors="coerce")
df_icp[VALUE_COL_ICP] = pd.to_numeric(df_icp["obs_value"], errors="coerce")

title_icp = df_icp["title_fr"].dropna().iloc[0] if df_icp["title_fr"].notna().any() else "Inflation (ICP)"

df_icp = (
    df_icp.dropna(subset=["date"])
          .sort_values("date")
          .set_index("date")
          [[VALUE_COL_ICP]]
)

print(df_icp.tail(10))

# ------------------------------------------------------------
# 4) Plot both time series (single axis, legend)
# ------------------------------------------------------------
df_plot = df_pai.join(df_icp, how="outer").sort_index()

plt.figure(figsize=(10, 5), dpi=150)

plt.plot(
    df_plot.index,
    df_plot[VALUE_COL],
    linewidth=1.5,
    label="PAI – Press-based inflation expectations"
)

plt.plot(
    df_plot.index,
    df_plot[VALUE_COL_ICP],
    linewidth=1.5,
    linestyle="--",
    label="ICP – Inflation (YoY)"
)

plt.xlabel("Date")
plt.ylabel("Percent")
plt.title("Inflation Expectations (PAI) and Observed Inflation (ICP)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()



# ------------------------------------------------------------
# Mincer–Zarnowitz regression (Inflation on Expectations)
#   π_t = α + β π_t^e + ε_t
# Focus: underreaction (β < 1) and unbiasedness (α=0, β=1)
# Inference: HAC (Newey–West) standard errors with 12 lags (monthly data)
# ------------------------------------------------------------

# 0) Balanced sample
df_mz = df_plot[["inflation_yoy", "pai_exp_infl"]].dropna().copy()
df_mz = df_mz.sort_index()

# 1) Estimate Mincer–Zarnowitz regression with HAC SE
# H0 for coefficients themselves is the usual t-test null (e.g., α=0, β=0),
# but our economic nulls are α=0 and β=1.
y = df_mz["inflation_yoy"]
X = sm.add_constant(df_mz["pai_exp_infl"])  # [const, π^e]

mz_model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 12})
print(mz_model.summary())

# Extract estimates
alpha_hat = mz_model.params["const"]
beta_hat  = mz_model.params["pai_exp_infl"]

# ------------------------------------------------------------
# Test A: Underreaction (one-sided slope test)
#   H0: β = 1
#   H1: β < 1   (underreaction)
# Conducted as a t-test using HAC standard error of β.
# ------------------------------------------------------------
se_beta = mz_model.bse["pai_exp_infl"]
t_beta = (beta_hat - 1.0) / se_beta

# Approximate p-value using Student t with df_resid
# (common in applied work; asymptotically standard normal is also used)
df_resid = int(mz_model.df_resid)
p_one_sided = stats.t.cdf(t_beta, df=df_resid)  # left-tail prob for H1: β < 1

print("\n=== Test A: Underreaction (β < 1) ===")
print("H0: β = 1")
print("H1: β < 1")
print(f"t-stat (HAC): {t_beta:.4f}")
print(f"one-sided p-value: {p_one_sided:.4f}")

# (Optional) Two-sided version for completeness:
p_two_sided = 2.0 * min(stats.t.cdf(t_beta, df=df_resid), 1.0 - stats.t.cdf(t_beta, df=df_resid))
print(f"two-sided p-value (for H1: β ≠ 1): {p_two_sided:.4f}")

# ------------------------------------------------------------
# Test B: Intercept (systematic bias)
#   H0: α = 0
#   H1: α ≠ 0
# Conducted as a t-test using HAC standard error of α.
# ------------------------------------------------------------
se_alpha = mz_model.bse["const"]
t_alpha = (alpha_hat - 0.0) / se_alpha
p_alpha_two_sided = 2.0 * (1.0 - stats.t.cdf(abs(t_alpha), df=df_resid))

print("\n=== Test B: Intercept Bias (α) ===")
print("H0: α = 0")
print("H1: α ≠ 0")
print(f"t-stat (HAC): {t_alpha:.4f}")
print(f"two-sided p-value: {p_alpha_two_sided:.4f}")

# ------------------------------------------------------------
# Test C: Joint unbiasedness (Wald test)
#   H0: α = 0 and β = 1
#   H1: at least one restriction fails
# Conducted as a Wald test with HAC covariance matrix.
# Asymptotically: χ² with 2 degrees of freedom.
# ------------------------------------------------------------
R = np.array([
    [1.0, 0.0],  # picks α
    [0.0, 1.0],  # picks β
])
r = np.array([0.0, 1.0])

wald = mz_model.wald_test((R, r))

print("\n=== Test C: Joint Unbiasedness (Wald) ===")
print("H0: α = 0 and β = 1")
print("H1: not(H0)")
print(wald)




# ============================================================
# Cointegration Analysis: Inflation and Expectations
# ============================================================


# ------------------------------------------------------------
# 1) Balanced sample (drop missing observations)
# ------------------------------------------------------------
df_coint = df_plot[["inflation_yoy", "pai_exp_infl"]].dropna().copy()
df_coint = df_coint.sort_index()

print("Sample period:",
      df_coint.index.min(), "to", df_coint.index.max())
print("Number of observations:", len(df_coint))

# ============================================================
# 2) Engle–Granger Test
# ------------------------------------------------------------
# Long-run regression:
#   π_t = α + β π_t^e + u_t
#
# H0: residuals have a unit root (no cointegration)
# H1: residuals are stationary (cointegration)
# ============================================================

y = df_coint["inflation_yoy"]
x = df_coint["pai_exp_infl"]

coint_stat, p_value, crit_values = coint(y, x)

print("\n=== Engle–Granger Cointegration Test ===")
print("H0: No cointegration (residual has unit root)")
print(f"Test statistic: {coint_stat:.4f}")
print(f"P-value: {p_value:.4f}")
print("Critical values:", crit_values)

# ============================================================
# 3) Johansen Rank Test (System Approach)
# ------------------------------------------------------------
# VECM representation:
#   ΔY_t = Π Y_{t-1} + Γ ΔY_{t-1} + ε_t
#
# H0: rank(Π) ≤ r
# H1: rank(Π) > r
# ============================================================

jres = coint_johansen(
    df_coint[["inflation_yoy", "pai_exp_infl"]].values,
    det_order=0,      # no deterministic trend in cointegration relation
    k_ar_diff=1       # 1 lag in differences (can be selected via information criteria)
)

print("\n=== Johansen Trace Test ===")
print("Trace statistics:", jres.lr1)
print("Critical values (90%, 95%, 99%):")
print(jres.cvt)

# ============================================================
# Johansen Trace Test — Interpretation
# ============================================================
# r = 0:
#   Trace statistic = 28.00
#   99% critical value = 19.93
#   → Reject H0: no cointegration.
#
# r ≤ 1:
#   Trace statistic = 8.38
#   99% critical value = 6.63
#   → Reject H0: at most one cointegration relation.
#
# Conclusion:
#   Estimated rank = 2 (in a 2-variable system).
#
# Economic implication:
#   Both inflation and inflation expectations behave as
#   stationary variables in levels (I(0)).
#   A VECM is therefore not required; a VAR in levels
#   is the appropriate framework.
#
# Always confirm with unit-root tests and proper lag selection.
# ============================================================


# ============================================================
# 4) VECM Estimation (if rank = 1)
# ------------------------------------------------------------
# Δπ_t = α (π_{t-1} - β π^e_{t-1}) + Γ ΔX_t + u_t
# ============================================================

vecm = VECM(
    df_coint[["inflation_yoy", "pai_exp_infl"]],
    k_ar_diff=1,
    coint_rank=1,     # set according to Johansen result
    deterministic="co"  # constant inside cointegration relation
)

vecm_res = vecm.fit()

print("\n=== VECM Estimation ===")
print(vecm_res.summary())



# ============================================================
# Coibion & Gorodnichenko (2012): Information Rigidity Test
#   FE_t = π_t - π_t^e
#   FE_t = c + λ FE_{t-1} + ε_t
#
# H0: λ = 0  (full information: no persistence in forecast errors)
# H1: λ > 0  (information rigidity: sluggish adjustment)
#
# Inference: HAC (Newey–West) standard errors with 12 lags (monthly data)
# ============================================================

# 1) Balanced sample
df_cg = df_plot[["inflation_yoy", "pai_exp_infl"]].dropna().copy()
df_cg = df_cg.sort_index()

# 2) Construct forecast error and its lag
df_cg["FE"] = df_cg["inflation_yoy"] - df_cg["pai_exp_infl"]
df_cg["FE_lag1"] = df_cg["FE"].shift(1)

# Drop first observation lost to lagging
df_cg = df_cg.dropna(subset=["FE", "FE_lag1"])

print("Sample period:",
      df_cg.index.min(), "to", df_cg.index.max())
print("Number of observations:", len(df_cg))

# 3) OLS with HAC standard errors
y = df_cg["FE"]
X = sm.add_constant(df_cg["FE_lag1"])

cg_model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 12})
print("\n=== Coibion–Gorodnichenko Regression ===")
print(cg_model.summary())

# 4) One-sided test for information rigidity: H1: λ > 0
lambda_hat = cg_model.params["FE_lag1"]
se_lambda = cg_model.bse["FE_lag1"]
t_lambda = (lambda_hat - 0.0) / se_lambda

df_resid = int(cg_model.df_resid)
p_one_sided = 1.0 - stats.t.cdf(t_lambda, df=df_resid)  # right tail for H1: λ > 0

print("\n=== Test: Information Rigidity ===")
print("H0: λ = 0 (no persistence in forecast errors)")
print("H1: λ > 0 (sluggish adjustment)")
print(f"lambda_hat: {lambda_hat:.4f}")
print(f"t-stat (HAC): {t_lambda:.4f}")
print(f"one-sided p-value: {p_one_sided:.4f}")

# (Optional) quick diagnostic: FE autocorrelation at lag 1
fe_autocorr = df_cg["FE"].autocorr(lag=1)
print(f"\nSample autocorr(FE, lag=1): {fe_autocorr:.4f}")


# ============================================================
# Uncertainty and Inflation Expectations
#   (1) Measure inflation uncertainty via GARCH volatility
#   (2) Baseline expectations regression: π^e_t on π_t and σ_t
#   (3) Dynamic version with lags and joint test on volatility terms
#   (4) Volatility × rigidity (CG-style interaction on forecast errors)
#   (5) System evidence: VAR and Granger causality (σ -> π^e)
# ============================================================



# ------------------------------------------------------------
# 1) Measuring inflation uncertainty: GARCH(1,1) on inflation
# ------------------------------------------------------------
# Model:
#   π_t = μ + ε_t,   ε_t = σ_t z_t
#   σ_t^2 = ω + α ε_{t-1}^2 + β σ_{t-1}^2
#
# Output:
#   infl_vol = σ_t   (conditional std dev, uncertainty proxy)
# ------------------------------------------------------------

df_u = df_plot[["inflation_yoy", "pai_exp_infl"]].dropna().copy()
df_u = df_u.sort_index()

pi = df_u["inflation_yoy"]

# GARCH volatility is defined for mean-zero residuals; we estimate a mean (constant) by default.
# dist="normal" matches the pedagogical setup; you can later switch to "t" for fat tails.
garch = arch_model(pi, mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
garch_res = garch.fit(disp="off")

df_u["infl_vol"] = garch_res.conditional_volatility          # σ_t
df_u["infl_var"] = df_u["infl_vol"] ** 2                     # σ_t^2 (optional)

print("\n=== GARCH(1,1) for Inflation ===")
print(garch_res.summary())

# (Optional) Plot volatility proxy
plt.figure(figsize=(10, 4), dpi=150)
plt.plot(df_u.index, df_u["infl_vol"], linewidth=1.5)
plt.title("Inflation Uncertainty Proxy: GARCH Conditional Volatility (σ_t)")
plt.xlabel("Date")
plt.ylabel("σ_t")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 2) Baseline expectations equation:
#   π^e_t = c + β π_t + γ σ_t + u_t
#
# Two-sided test:
#   H0: γ = 0
#   H1: γ ≠ 0
#
# One-sided (directional) test:
#   H0: γ = 0
#   H1: γ > 0   (uncertainty raises expectations)
#
# Inference: HAC (Newey–West), 12 lags (monthly)
# ------------------------------------------------------------

y = df_u["pai_exp_infl"]
X = sm.add_constant(df_u[["inflation_yoy", "infl_vol"]])

exp_baseline = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 12})
print("\n=== Baseline Expectations Regression ===")
print("Model: pi_e ~ const + pi + infl_vol")
print("Test of interest: H0: gamma(vol) = 0")
print(exp_baseline.summary())

gamma_hat = exp_baseline.params["infl_vol"]
se_gamma = exp_baseline.bse["infl_vol"]
t_gamma = gamma_hat / se_gamma
df_resid = int(exp_baseline.df_resid)

# Two-sided p-value (H1: γ ≠ 0)
p_gamma_two_sided = 2.0 * (1.0 - stats.t.cdf(abs(t_gamma), df=df_resid))

# One-sided p-value (H1: γ > 0)
# Right-tail probability under H0
p_gamma_one_sided = 1.0 - stats.t.cdf(t_gamma, df=df_resid)

print("\n=== Test: Volatility effect in baseline regression ===")
print("H0: γ = 0")
print("H1 (two-sided): γ ≠ 0")
print("H1 (one-sided): γ > 0")
print(f"gamma_hat: {gamma_hat:.4f}")
print(f"t-stat (HAC): {t_gamma:.4f}")
print(f"two-sided p-value: {p_gamma_two_sided:.4f}")
print(f"one-sided p-value (γ > 0): {p_gamma_one_sided:.4f}")

# Two-sided test:
#   Reject H0: γ = 0  → volatility affects expectations.
#
# One-sided test (H1: γ > 0):
#   Fail to reject H0 → no evidence that higher volatility
#   raises inflation expectations.


# ------------------------------------------------------------
# 3) Dynamic expectations equation (pedagogical extension):
#   π^e_t = c + Σ φ_j π^e_{t-j} + Σ β_j π_{t-j} + Σ γ_j σ_{t-j} + u_t
#
# Joint test:
#   H0: γ_0 = γ_1 = ... = γ_p = 0
# ------------------------------------------------------------

p = 3  # number of lags (keep small for lecture; can be selected via AIC/BIC)

df_dyn = df_u.copy()
for j in range(1, p + 1):
    df_dyn[f"pi_e_l{j}"] = df_dyn["pai_exp_infl"].shift(j)
    df_dyn[f"pi_l{j}"] = df_dyn["inflation_yoy"].shift(j)
    df_dyn[f"vol_l{j}"] = df_dyn["infl_vol"].shift(j)

# include contemporaneous π_t and σ_t as "lag 0"
reg_cols = ["inflation_yoy", "infl_vol"] \
           + [f"pi_e_l{j}" for j in range(1, p + 1)] \
           + [f"pi_l{j}" for j in range(1, p + 1)] \
           + [f"vol_l{j}" for j in range(1, p + 1)]

df_dyn = df_dyn.dropna(subset=["pai_exp_infl"] + reg_cols)

y_dyn = df_dyn["pai_exp_infl"]
X_dyn = sm.add_constant(df_dyn[reg_cols])

exp_dyn = sm.OLS(y_dyn, X_dyn).fit(cov_type="HAC", cov_kwds={"maxlags": 12})

print("\n=== Dynamic Expectations Regression ===")
print(f"Included lags p={p}")
print(exp_dyn.summary())

# Joint Wald test for all volatility terms (current + lags)
vol_terms = ["infl_vol"] + [f"vol_l{j}" for j in range(1, p + 1)]
param_names = exp_dyn.params.index.tolist()

R = np.zeros((len(vol_terms), len(param_names)))
for i, name in enumerate(vol_terms):
    R[i, param_names.index(name)] = 1.0
r0 = np.zeros(len(vol_terms))

wald_vol = exp_dyn.wald_test((R, r0))

print("\n=== Joint Volatility Test (Dynamic Spec) ===")
print("H0: all volatility coefficients = 0 (gamma_0=...=gamma_p=0)")
print(wald_vol)

# ------------------------------------------------------------
# 4) Volatility and information rigidity (CG interaction)
#   FE_t = π_t - π^e_t
#   FE_t = a + λ FE_{t-1} + δ σ_t + κ (FE_{t-1}×σ_t) + η_t
#
# Key test:
#   H0: κ = 0  (volatility does not amplify rigidity)
#   H1: κ > 0
# ------------------------------------------------------------

df_int = df_u.copy()
df_int["FE"] = df_int["inflation_yoy"] - df_int["pai_exp_infl"]
df_int["FE_l1"] = df_int["FE"].shift(1)
df_int = df_int.dropna(subset=["FE", "FE_l1", "infl_vol"])

df_int["FE_l1_x_vol"] = df_int["FE_l1"] * df_int["infl_vol"]

y_int = df_int["FE"]
X_int = sm.add_constant(df_int[["FE_l1", "infl_vol", "FE_l1_x_vol"]])

cg_vol = sm.OLS(y_int, X_int).fit(cov_type="HAC", cov_kwds={"maxlags": 12})
print("\n=== CG with Volatility Interaction ===")
print("Model: FE ~ const + FE_{t-1} + vol + FE_{t-1}×vol")
print("Key test: H0: kappa(interaction)=0 vs H1: kappa>0")
print(cg_vol.summary())

kappa_hat = cg_vol.params["FE_l1_x_vol"]
se_kappa = cg_vol.bse["FE_l1_x_vol"]
t_kappa = kappa_hat / se_kappa
df_resid = int(cg_vol.df_resid)
p_kappa_one_sided = 1.0 - stats.t.cdf(t_kappa, df=df_resid)  # right tail for H1: κ > 0

print("\n=== Test: Volatility amplifies rigidity (interaction) ===")
print("H0: κ = 0")
print("H1: κ > 0")
print(f"kappa_hat: {kappa_hat:.4f}")
print(f"t-stat (HAC): {t_kappa:.4f}")
print(f"one-sided p-value: {p_kappa_one_sided:.4f}")

# ------------------------------------------------------------
# 5) System evidence: VAR and Granger causality (σ -> π^e)
#   Z_t = [π^e_t, π_t, σ_t]'
#
# Granger test:
#   H0: lags of σ_t do not enter π^e equation
# ------------------------------------------------------------

df_var = df_u[["pai_exp_infl", "inflation_yoy", "infl_vol"]].dropna().copy()

var_maxlags = 12
var_model = VAR(df_var)

# Pick lag length by BIC (robust pedagogical default for monthly)
lag_sel = var_model.select_order(maxlags=var_maxlags)
p_var = lag_sel.selected_orders.get("bic", 2)  # fallback to 2 if missing

var_res = var_model.fit(p_var)

print("\n=== VAR Results ===")
print(f"Lag selection (max {var_maxlags}) -> chosen p = {p_var}")
print(var_res.summary())

# Granger causality test: infl_vol does not Granger-cause expectations (pai_exp_infl)
gc = var_res.test_causality(caused="pai_exp_infl", causing=["infl_vol"], kind="f")

print("\n=== Granger Causality Test: volatility -> expectations ===")
print("H0: lags of infl_vol do not help predict pai_exp_infl")
print(gc.summary())
