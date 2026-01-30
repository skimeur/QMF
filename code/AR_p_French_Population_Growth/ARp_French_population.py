#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantitative Methods in Finance (QMF)

AR(p) Model for French Population Growth
----------------------------------------
Beginner-level pandas exercise illustrating autoregressive modeling
of French population growth rates.

Content:
- Data cleaning and time indexing
- Unit root testing and growth-rate transformation
- AR(1) estimation via OLS and maximum likelihood
- Bias and consistency discussion
- Seasonality removal
- Higher-order AR(p) models and lag selection (AIC/BIC)
- Residual diagnostics (PACF, Ljung–Box test)

Author: Eric Vansteenberghe
Year: 2026
"""


import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2
from scipy.optimize import minimize
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model  import ARIMA
import matplotlib.pyplot as plt

# to plot, set ploton to ploton to 1
ploton = False

# change the working directory
os.chdir('//Users/skimeur/Mon Drive/QMF/')

# =============================================================================
# 1) Load and clean French population data
# =============================================================================
DATA_PATH = "data/Valeurs.csv"

df = (
    pd.read_csv(
        DATA_PATH,
        sep=";",
        encoding="latin1",
        skiprows=[0, 1, 2],
        header=None,
        names=["Year", "Month", "Population"],
        dtype={"Year": int, "Month": int, "Population": str},
    )
    .iloc[::-1]  # file is often reverse-chronological
    .assign(
        Date=lambda x: pd.to_datetime(
            x["Year"].astype(str) + x["Month"].astype(str).str.zfill(2),
            format="%Y%m",
        )
    )
    .set_index("Date")
    .drop(columns=["Year", "Month"])
)

# Clean population column (strip spaces, convert to numeric)
df["Population"] = (
    df["Population"]
    .astype(str)
    .str.replace(" ", "", regex=False)
    .pipe(pd.to_numeric, errors="coerce")
)

# Basic sanity checks
if df["Population"].isna().any():
    n_bad = int(df["Population"].isna().sum())
    raise ValueError(f"{n_bad} invalid Population values after parsing in {DATA_PATH}.")

df = df.sort_index()

# =============================================================================
# Optional diagnostics plots
# =============================================================================
if ploton:
    ax = df.plot(
        title="French Population Level",
        legend=False,
        figsize=(8, 4),
    )
    ax.set_ylabel("Population")

    ax = (df.diff()).plot(
        title="Monthly Change in French Population",
        legend=False,
        figsize=(8, 4),
    )
    ax.set_ylabel("Δ Population")



# =============================================================================
# Unit root testing: Augmented Dickey–Fuller (ADF)
# =============================================================================
# The ADF test evaluates the null hypothesis H0: the series has a unit root (I(1)).
# Rejection of H0 suggests stationarity.
#
# Regression specifications:
#  - 'n'  : no constant, no trend (use only if theory implies zero mean)
#  - 'c'  : constant (default; stationary around a non-zero mean)
#  - 'ct' : constant + linear trend (trend-stationary alternative)
#
# In practice, results should be compared across specifications.

def adf_report(series, label, regressions=("n", "c", "ct")):
    """Run ADF tests under multiple deterministic specifications."""
    print(f"\nADF test results — {label}")
    for reg in regressions:
        stat, pval, _, _, crit, _ = adfuller(series, regression=reg, autolag="AIC")
        print(
            f"  regression='{reg}': "
            f"ADF stat = {stat: .3f}, p-value = {pval: .3f}, "
            f"5% crit = {crit['5%']: .3f}"
        )

# ADF on population level
adf_report(df["Population"], label="Population level")

# ADF on first differences (growth approximation)
adf_report(df["Population"].diff().dropna(), label="First difference")

# =============================================================================
# Simple outlier treatment (single positive outlier)
# =============================================================================
# Monthly population growth rate
dx = (df - df.shift(1)) / df.shift(1)

# Identify the largest growth observation
outlier_value = dx["Population"].max()

# Add month information
dx["Month"] = dx.index.month

# The outlier occurs in January (Month == 1)
# Replace it with the average January growth rate (excluding the outlier)
dx.loc[dx["Population"] == outlier_value, "Population"] = (
    dx.loc[(dx["Population"] != outlier_value) & (dx["Month"] == 1), "Population"].mean()
)

# =============================================================================
# Optional visualization: effect of outlier treatment
# =============================================================================
if ploton:
    ax = (df.diff()).plot(
        title="Monthly Population Change (Raw)",
        legend=False,
        figsize=(8, 4),
    )
    ax.set_ylabel("Δ Population")

    ax = dx["Population"].plot(
        title="Monthly Population Growth (Outlier Corrected)",
        legend=False,
        figsize=(8, 4),
        secondary_y=True
    )
    ax.set_ylabel("Growth rate")


# =============================================================================
# Demean population growth rate
# =============================================================================
dx["Population_demeaned"] = dx["Population"] - dx["Population"].mean()

# Optional visualization
if ploton:
    ax = dx[["Population", "Population_demeaned"]].plot(
        title="Population Growth: Raw vs. Demeaned",
        figsize=(8, 4),
    )
    ax.set_ylabel("Growth rate")

# =============================================================================

# Remove deterministic month-of-year seasonality (monthly means)
# =============================================================================

dx["season_mean"] = np.nan
for m in range(1, 13):
    dx.loc[dx["Month"] == m, "season_mean"] = dx.loc[dx["Month"] == m, "Population_demeaned"].mean()

if ploton:
    ax = dx["season_mean"].plot(
        title="Estimated Seasonality (Month-of-Year Mean)",
        legend=False,
        figsize=(8, 4),
    )
    ax.set_ylabel("Seasonal mean")


dx["Population_deseasonalized"] = dx["Population_demeaned"] - dx["season_mean"]

if ploton:
    ax = dx["Population_deseasonalized"].plot(
        title="Demeaned and Deseasonalized Population Growth",
        legend=False,
        figsize=(8, 4),
    )
    ax.set_ylabel("Growth rate")


# =============================================================================
# Final stationary series used for AR modeling
# =============================================================================
ytild = dx["Population_deseasonalized"].dropna()

# Optional visualization
if ploton:
    ax = ytild.plot(
        title="Demeaned, Deseasonalized Population Growth Rate",
        legend=False,
        figsize=(8, 4),
    )
    ax.set_ylabel("Growth rate")

# ADF test on demeaned growth rate
adf_report(ytild, label="Demeaned, Deseasonalized population growth rate")



# =============================================================================
# AR(1) on demeaned growth + simple seasonal adjustment
# =============================================================================

# 1) Fit AR(1) on demeaned growth rate
ar1 = ARIMA(ytild, order=(1, 0, 0), trend="n").fit()
print(ar1.summary())

# =============================================================================
# Inspecting a fitted ARIMA result (statsmodels)
# =============================================================================
# `dir(ar1)` is exhaustive (and includes many internal/private names).
# A short list of the most common, high-value attributes and methods.

KEY_ITEMS = {
    # Model selection / fit quality
    "aic": "Akaike Information Criterion (lower is better, same data).",
    "bic": "Bayesian Information Criterion (stronger penalty for complexity).",
    "llf": "Log-likelihood at the optimum.",
    # Parameters and inference
    "params": "Estimated parameters (pandas Series).",
    "bse": "Standard errors of parameters.",
    "pvalues": "p-values for parameters.",
    "conf_int": "Confidence intervals for parameters.",
    # Fitted objects
    "fittedvalues": "In-sample fitted values.",
    "resid": "In-sample residuals.",
    # Forecasting
    "forecast": "Point forecasts for future periods.",
    "get_forecast": "Forecast object with confidence intervals.",
    "predict": "In-sample / out-of-sample predictions.",
    # Diagnostics / plots
    "plot_diagnostics": "Standard diagnostics panel (residual checks).",
    "test_serial_correlation": "Residual serial correlation test (Ljung–Box style).",
    "test_normality": "Residual normality test.",
}

print("Commonly used items on a fitted ARIMA result:")
for k, v in KEY_ITEMS.items():
    print(f"  - {k}: {v}")

# ---- Illustration: extract a few concrete elements ----
print("\nParameter estimates:")
print(ar1.params)

print("\nAR coefficient(s) (phi):")
print(ar1.arparams)  # for AR(1), this prints a 1-element array

# Forecast with a 95% confidence interval for the next 12 months
H = 12
fc = ar1.get_forecast(steps=H)
fc_mean = fc.predicted_mean
fc_ci = fc.conf_int(alpha=0.05)

print(f"\nNext {H} forecasted values (mean):")
print(fc_mean)

print(f"\nNext {H} forecast 95% confidence intervals:")
print(fc_ci)

# Optional plot
if ploton:
    ax = ytild.plot(title="Demeaned Population Growth: AR(1) Fit and Forecast", figsize=(9, 4))
    ar1.fittedvalues.plot(ax=ax, linestyle="--")
    fc_mean.plot(ax=ax)
    ax.fill_between(fc_ci.index, fc_ci.iloc[:, 0], fc_ci.iloc[:, 1], alpha=0.2)
    ax.set_ylabel("Growth rate")
    ax.legend(["Observed", "Fitted", "Forecast", "95% CI"])


# =============================================================================
# OLS estimation of an AR(p) model (illustration with p up to 3)
# =============================================================================
# We estimate an AR(1) by OLS on the demeaned series:
#   y_t = theta * y_{t-1} + e_t
# Note: OLS in dynamic models can be biased in small samples, but is consistent
# under standard stationarity conditions.

# Build a regression DataFrame with lags (up to 3 for later AR(3) exercises)
dftild = pd.concat(
    [ytild, ytild.shift(1), ytild.shift(2), ytild.shift(3)],
    axis=1
).dropna()
dftild.columns = ["yt", "ytminus1", "ytminus2", "ytminus3"]

# AR(1) by OLS, no intercept (ytild is demeaned)
ols_ar1 = smf.ols("yt ~ ytminus1 - 1", data=dftild).fit()

print(ols_ar1.summary())

# OLS estimate of theta (AR(1) coefficient)
theta_ols = float(ols_ar1.params["ytminus1"])
print(f"\nOLS AR(1) estimate: theta = {theta_ols:.4f}")

# Residuals (useful later for diagnostics and confidence intervals)
resid_ols = ols_ar1.resid

if ploton:
    ax = resid_ols.plot(
        title="OLS AR(1) Residuals",
        legend=False,
        figsize=(8, 3),
    )
    ax.set_ylabel("Residual")

# Next steps (lecture exercises):
# - Compare OLS vs. maximum likelihood (ARIMA) estimates
# - Check residual serial correlation (e.g., Ljung–Box)
# - Build forecasts and confidence intervals using residual distribution / simulation

# =============================================================================
# Maximum Likelihood Estimation (MLE) for AR(1) under Gaussian errors
# =============================================================================
# Model (demeaned series):
#   y_t = theta * y_{t-1} + eps_t,   eps_t ~ N(0, sigma^2)
# Stationary initialization:
#   y_0 ~ N(0, sigma^2 / (1 - theta^2))  for |theta| < 1
#
# We minimize the *negative* log-likelihood (up to an additive constant).

def neg_loglike_ar1(theta, y):
    """Negative log-likelihood of AR(1) with Gaussian errors and stationary y0."""
    theta = float(theta)  # minimize passes an array-like

    # Enforce stationarity for the initialization formula (simple penalty)
    if abs(theta) >= 0.999:
        return 1e10

    # One-step-ahead innovations (skip the first NaN due to shift)
    e = (y - theta * y.shift(1)).dropna()

    # Innovation variance estimate given theta
    sigma2 = float(e.var(ddof=0))
    if sigma2 <= 0 or not np.isfinite(sigma2):
        return 1e10

    T = len(y.dropna())

    # Contribution of y0 under stationary distribution
    y0 = float(y.dropna().iloc[0])
    var0 = sigma2 / (1.0 - theta**2)

    nll = 0.5 * (
        T * np.log(2 * np.pi)
        + np.log(var0) + (y0**2) / var0
        + (T - 1) * np.log(sigma2)
        + (e**2).sum() / sigma2
    )
    return float(nll)

# Minimize negative log-likelihood
theta_start = 0.8
mle_ar1 = minimize(
    neg_loglike_ar1,
    x0=[theta_start],
    args=(ytild,),
    method="Nelder-Mead",
)

theta_mle = float(mle_ar1.x[0])
print(f"MLE AR(1) estimate: theta = {theta_mle:.4f}")
print(f"MLE converged: {mle_ar1.success} | iterations: {mle_ar1.nit}")



# =============================================================================
# AR(3): OLS estimation, PACF, and AR order selection
# =============================================================================

# 1) AR(3) by OLS (no intercept since ytild is demeaned)
ols_ar3 = smf.ols("yt ~ ytminus1 + ytminus2 + ytminus3 - 1", data=dftild).fit()
print(ols_ar3.summary())

# 2) PACF plot (useful to visually assess AR order)
if ploton:
    sm.graphics.tsa.plot_pacf(ytild, lags=10)
    plt.tight_layout()

# 3) Select AR order using information criteria (ARMA with max_ma=0 => AR only)
# Note: this is a quick automated selection for teaching purposes.
order_sel = sm.tsa.arma_order_select_ic(
    ytild,
    ic=["aic", "bic"],
    trend="n",
    max_ar=10,
    max_ma=0,
)

p_aic = order_sel.aic_min_order[0]
p_bic = order_sel.bic_min_order[0]
print(f"\nSelected AR order by AIC: p = {p_aic}")
print(f"Selected AR order by BIC: p = {p_bic}")

# 4) Fit AR(3) by maximum likelihood (ARIMA with q=0, no trend)
ar3 = ARIMA(ytild, order=(3, 0, 0), trend="n").fit()
print(ar3.summary())

# Optional: residual plot + save to disk
if ploton:
    ax = ar3.resid.plot(
        title="AR(3) Residuals (MLE fit)",
        legend=False,
        figsize=(8, 3),
    )
    ax.set_ylabel("Residual")

    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig("fig/AR3_resid.pdf")



# =============================================================================
# Ljung–Box test for residual serial correlation
# =============================================================================
# H0: no autocorrelation in residuals up to lag h
# For AR(p) residuals, the asymptotic chi-square degrees of freedom is (h - p).

def autocorr(x, lag):
    """Sample autocorrelation at a given lag."""
    x = np.asarray(x)
    return np.corrcoef(x[:-lag], x[lag:])[0, 1]

def ljung_box(resid, h, p=0):
    """
    Ljung–Box test up to lag h for residuals `resid`.
    Returns: (Qstat, df, pvalue)
    """
    x = pd.Series(resid).dropna().values
    T = len(x)

    # Ljung–Box Q statistic
    q = 0.0
    for k in range(1, h + 1):
        rho_k = autocorr(x, k)
        q += (rho_k**2) / (T - k)
    Qstat = T * (T + 2) * q

    # Degrees of freedom adjustment for AR(p)
    df = max(h - p, 1)
    pvalue = 1.0 - chi2.cdf(Qstat, df=df)
    return Qstat, df, pvalue

# Example: test AR(3) residuals up to lag 10
maxlag = 10
Qstat, df_used, pval = ljung_box(ar3.resid, h=maxlag, p=3)

dfLB = pd.DataFrame(
    {"AR3": [Qstat, pval]},
    index=["Q-statistic", "p-value"],
)
print(dfLB)
print(f"\nLjung–Box df used = {df_used} (h={maxlag}, p=3)")


