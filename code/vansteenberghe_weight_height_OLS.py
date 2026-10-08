#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QMF 2026 — Linear regression on weight and height (OLS, diagnostics, and ML benchmark)

This script accompanies Section \\ref{sec:weightheightregression} of the QMF lecture notes.
It illustrates, on a simple cross-sectional dataset, how to (i) check basic association
between two variables, (ii) estimate linear regressions by Ordinary Least Squares (with
and without an intercept), (iii) reproduce OLS slope formulas manually, (iv) run standard
diagnostics for the Classical Linear Regression Model, and (v) compare parametric fits
with a nonparametric machine-learning benchmark (Random Forest).

Data and variables
------------------
Input file:  data/replication_final.dta  (read via pandas.read_stata)

Extracted columns:
- v002 : household number (ID for deduplication)
- v012 : respondent age (used for deduplication)
- v024 : state (not used in the baseline regression, kept for extensions)
- v437 : women's weight in 0.1 kg units
- v438 : women's height in millimeters

Pre-processing choices:
- Keep unique observations by (v002, v012).
- Drop obvious outliers (tail trimming)

Workflow overview
-----------------
1) Association checks:
   - Pearson correlation and Spearman rank correlation.
   - A chi-square test is shown in the code, but note that chi-square is designed for
     categorical contingency tables; correlation measures are the appropriate default.

2) OLS estimation:
   - OLS with intercept:      v437 ~ v438
   - OLS without intercept:   v437 ~ v438 - 1
   The no-intercept specification forces E[y|x=0]=0 and can mechanically inflate R²;
   it is included to illustrate why this restriction is usually unjustified.

3) Manual verification:
   - Compute the closed-form OLS slope with and without an intercept and verify that
     it matches statsmodels estimates up to numerical tolerance.

4) Visualization (optional; controlled by ploton):
   - Scatter plot and fitted lines (with/without intercept), with unit conversions:
     height mm → meters; weight 0.1kg → kg.

5) Classical OLS diagnostics (for the intercept model):
   - Functional form: Ramsey RESET (misspecification / neglected nonlinearity).
   - No perfect multicollinearity: Sxx = sum((x - xbar)^2) > 0 (mechanical check).
   - Homoskedasticity: Breusch–Pagan and White tests + robust HC1 inference if needed.
   - Notes are included on what is and is not testable from the sample (random sampling,
     exogeneity).

5b) Companion block for the lecture notes (runs after the diagnostics):
   - Sample composition (v000) and what the DHS missing-value code 9999 does to OLS.
   - Weight in kg and height in cm; OLS with and without intercept, centred height.
   - Algebraic properties of OLS, reverse regression and regression to the mean.
   - Classical and heteroskedasticity-robust (HC0-HC3) standard errors, prediction.
   - Monte Carlo sampling distribution of the slope (our women as the population).
   - Residual spread by height quintile, Breusch-Pagan and White tests, leverage,
     influence and residual normality. Figures fig/weight_height_binned_means.pdf
     and fig/weight_height_OLS_sampling.pdf when ploton = True.

6) Alternative specifications:
   - Log-linear regression: log(weight) on height.
   - Log(1+y) regression: log(1+weight) on height (useful when many zeros exist).
   - Poisson GLM as an illustrative link-function analogue to log-linear models.

7) Machine-learning benchmark:
   - RandomForestRegressor fit to predict log(weight) from height.
   - Simple summary (R², RMSE, feature importance) and a finite-difference style
     “average sensitivity” approximation for interpretability.

Outputs
-------
- Printed regression tables and diagnostic test results in the console.
- Optional figures saved under fig/ when ploton = True.

@author: Eric Vansteenberghe
"""


from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import statsmodels.formula.api as smf # for linear regressions
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import os
from statsmodels.stats.diagnostic import (
    het_breuschpagan,
    het_white,
    linear_reset,
)

ploton = False

# We set the working directory
os.chdir('/Users/skimeur/Mon Drive/QMF')

df = pd.read_stata("data/replication_final.dta", convert_categoricals=False, columns= ["v002","v012","v024","v437","v438"])

# v002 household number
# v012 current age (respondent)
# v024 state
# v437 women's weight in .1 of kg
# v438 women's height in mm

# keep only unique observations
df = df.drop_duplicates(subset=["v002","v012"])

# we seem to have unexpected outliers
# in this study, we are not focusing on the tail hence we drop outliers
df = df.loc[(df.v437<1600)&(df.v438<2000)&(df.v438>1200),:]


#%% Linear regression

# Scatter plot
if ploton:
    ax = df.plot.scatter(x= "v438",y="v437")
    ax.set_xlabel("height")
    ax.set_ylabel("weight")
    fig = ax.get_figure()
    fig.savefig('fig/Indian_height_weight_scatter.pdf')

# compute the correlation
print('Weight and Height correlation:', round(100 * df.loc[:,['v437','v438']].corr().iloc[0,1]), '%')
pearson_r = df.loc[:, ['v437', 'v438']].corr().iloc[0, 1]  # used in the figure below

# a linear regression with an intercept
modelOLS = smf.ols('v437 ~  v438',data = df).fit()
modelLaTeX = modelOLS.summary().as_latex()
print(modelOLS.summary())

# a linear regression without an intercept
modelOLS_no_intercept = smf.ols('v437 ~  v438 - 1',data = df).fit()
modelLaTeX_no_intercept = modelOLS_no_intercept.summary().as_latex()
print(modelOLS_no_intercept.summary())
# Look, we get a very high R-square with no intercept!!!
# But be confident that this regression specification is false,
# we force the line to pass through (0,0) for no good reason

# manual computation of the beta for the model with no constant
betacalcule = (df.v437 * df.v438).sum() / (df.v438**2).sum()

# make sure that our manual computation is close enough to the estimation we get in python
np.abs(betacalcule - modelOLS_no_intercept.params['v438']) < 10**(-5)

# manual computation of the beta for the model with a constant
beta1calcule = (df.v438 * (df.v437 - df.v437.mean())).sum() / (df.v438 * (df.v438-df.v438.mean())).sum()

# make sure that our manual computation is close enough to the estimation we get in python
np.abs(beta1calcule - modelOLS.params.iloc[1]) < 10**(-5)

beta0calcule = df.v437.mean() - beta1calcule * df.v438.mean()

# manual computation of the beta for the model with a constant
(beta0calcule - modelOLS.params.iloc[0]) < 10**(-5)

# Visual of both regressions
# we tae the constant and the estimated beta
alpha = modelOLS.params.iloc[0]
beta = modelOLS.params.iloc[1]
beta2 = modelOLS_no_intercept.params.iloc[0]


# plot the data
stepsize = 0.001
x = np.arange(df.v438.min(),1.1*df.v438.max(),stepsize)

if ploton:

    # ------------------------------------------------------------
    # Unit conversion
    # height: mm -> m
    # weight: 0.1 kg -> kg
    # ------------------------------------------------------------
    height_m = df["v438"] / 10**3
    weight_kg = df["v437"] / 10

    # X grid (in meters)
    x_min, x_max = height_m.min(), height_m.max()
    x_grid_m = np.linspace(x_min, x_max, 250)

    # Convert grid back to original units for prediction
    x_grid_cm = x_grid_m * 1000

    # Fitted values (original model units → convert to kg)
    y_hat_const = (
        modelOLS.params["Intercept"]
        + modelOLS.params["v438"] * x_grid_cm
    ) / 10

    y_hat_noconst = (
        modelOLS_no_intercept.params["v438"] * x_grid_cm
    ) / 10

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=150)

    ax.scatter(
        height_m,
        weight_kg,
        s=10,
        alpha=0.25,
        edgecolors="none",
        rasterized=True,
        label="Data",
    )

    ax.plot(x_grid_m, y_hat_const, linewidth=2.0, label="OLS with intercept")
    ax.plot(x_grid_m, y_hat_noconst, linewidth=2.0, linestyle="--",
            label="OLS w/o intercept")

    ax.set_xlabel("Height (m)")
    ax.set_ylabel("Weight (kg)")
    ax.set_title("Weight–Height relationship (kg vs meters)")

    txt = (
        f"R² (with intercept)   = {modelOLS.rsquared:.3f}\n"
        f"R² (no intercept)     = {modelOLS_no_intercept.rsquared:.3f}\n"
        f"Pearson corr          = {pearson_r:.3f}"
    )

    ax.text(
        0.02, 0.98, txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3",
                  facecolor="white", alpha=0.8, linewidth=0.5),
    )

    ax.legend(frameon=True)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig("fig/Indian_height_weight_scatter_fit.pdf",
                bbox_inches="tight")
    plt.show()


#%% Quantile regression, a simple intro
# https://www.statsmodels.org/devel/examples/notebooks/generated/quantile_regression.html

modmedian = smf.quantreg("v437 ~  v438", df)
resmedian = modmedian.fit(q=0.5)
print(resmedian.summary())

# Height grid in the original units (mm)
grid = pd.DataFrame({"v438": np.linspace(df.v438.min(), df.v438.max(), 100)})

plt.figure(figsize=(7, 5))
plt.scatter(df.v438 / 1000, df.v437 / 10,
            s=10, alpha=0.2, color="grey", label="Data")
plt.plot(grid.v438 / 1000, modelOLS.predict(grid) / 10,
         color="blue", label="OLS")
plt.plot(grid.v438 / 1000, resmedian.predict(grid) / 10,
         color="red", linestyle="--", label="Median regression")

plt.xlabel("Height (m)")
plt.ylabel("Weight (kg)")
plt.legend()
plt.tight_layout()
plt.show()

# the median or mean approach are "relatively similar", so we can move on with an OLS

#%% ------------------------------------------------------------
# Diagnostics for Classical Linear Regression Assumptions (OLS)
# ------------------------------------------------------------
# What can be tested?
# A1 (linearity in parameters): not "tested" directly, but we can test functional form
#     with Ramsey RESET (misspecification / neglected nonlinearity).
# A2 (random sampling): cannot be tested with the sample alone; discuss as design assumption.
#     We can still check duplicate structure, clustering, and obvious selection patterns.
# A3 (no perfect multicollinearity): can be checked mechanically (variance of x > 0),
#     and numerically via condition number / rank (in multivariate case).
# A4 (exogeneity): not testable without instruments / experiments. However:
#     - in simple OLS with intercept, residuals are orthogonal to regressors by construction,
#       so corr(resid, x) = 0 is NOT a test of exogeneity.
#     - you can do placebo/robustness checks (add controls, fixed effects) but not a proof.
# A5 (homoskedasticity): testable with Breusch–Pagan / White.
# Also useful in classic lectures:
#     - Normality of errors (for exact small-sample t/F): Jarque–Bera.
#     - Influential points / leverage: Cook's distance, hat values.
#     - In cross-section, "autocorrelation" is usually not relevant, but if data are ordered
#       (e.g., time), then BG/DW could be used.

# Convenience objects
resid = modelOLS.resid
y = df["v437"].astype(float)
x = df["v438"].astype(float)

print(modelOLS.summary())

print('s: ', np.sqrt((modelOLS.resid**2).sum()/(len(modelOLS.resid)-2)))
print('weight standard deviation in the sample: ', df["v437"].std())

#%% ------------------------------------------------------------
# (A1) Functional form / linearity (Ramsey RESET)
# ------------------------------------------------------------
# H0 (Ramsey RESET): the model is correctly specified (no omitted nonlinear terms; added powers have zero coefficients)

# RESET tests whether adding powers of fitted values improves the model;
# rejection suggests neglected nonlinearity (or other misspecification).
reset = linear_reset(modelOLS, power=2, use_f=True)  # power=2 or 3 are common
print("\n(A1) Ramsey RESET (power=2):")
print(f"  F-stat = {reset.fvalue:.4f}, p-value = {reset.pvalue:.4g}")

# Optional: compare with a quadratic specification as a pedagogical illustration
model_quad = smf.ols("v437 ~ v438 + I(v438**2)", data=df).fit()
print("\nQuadratic augmentation (illustrative):")
print(f"  R2 linear: {modelOLS.rsquared:.4f}  |  R2 quadratic: {model_quad.rsquared:.4f}")
print(f"  p-value on I(v438**2): {model_quad.pvalues.get('I(v438 ** 2)', np.nan):.4g}")

print(model_quad.summary())

#%% ------------------------------------------------------------
# (A2) Random sampling: cannot be tested
# ------------------------------------------------------------

#%% ------------------------------------------------------------
# (A3) No perfect multicollinearity: check Var(x) > 0 and design matrix diagnostics
# ------------------------------------------------------------
Sxx = ((x - x.mean()) ** 2).sum()
print("\n(A3) No perfect multicollinearity:")
print(f"  Sxx = sum((x - xbar)^2) = {Sxx:.4e}  (must be > 0)")

#%% ------------------------------------------------------------
# (A4) Exogeneity E(e|x)=0: not testable, corr(resid, x) = 0 holds mechanically in OLS with intercept


#%% ------------------------------------------------------------
# (A5) Homoskedasticity: Breusch–Pagan and White tests
# ------------------------------------------------------------
# H0 (Breusch–Pagan and White): Var(e_i | X) = sigma^2  (homoskedasticity)
# H1: Var(e_i | X) depends on X (heteroskedasticity)
# BP requires exog matrix; include constant
exog = modelOLS.model.exog  # already includes intercept
bp_lm, bp_lmpval, bp_f, bp_fpval = het_breuschpagan(resid, exog)
print("\n(A5) Homoskedasticity tests:")
print("  Breusch–Pagan:")
print(f"    LM stat = {bp_lm:.4f}, p-value = {bp_lmpval:.4g}")
print(f"    F  stat = {bp_f:.4f}, p-value = {bp_fpval:.4g}")

white_lm, white_lmpval, white_f, white_fpval = het_white(resid, exog)
print("  White:")
print(f"    LM stat = {white_lm:.4f}, p-value = {white_lmpval:.4g}")
print(f"    F  stat = {white_f:.4f}, p-value = {white_fpval:.4g}")

# If heteroskedasticity suspected, show robust (HC1) standard errors
modelOLS_HC1 = modelOLS.get_robustcov_results(cov_type="HC1")
print("\nHeteroskedasticity-robust inference (HC1):")
print(modelOLS_HC1.summary())

#%% ============================================================
# COMPANION TO THE LECTURE NOTES (section "Linear regression on weight and height")
# Every number quoted in that section is printed by this block or by the cells
# above (RESET, quadratic R2). Units: weight in kg (v437 / 10), height in cm
# (v438 / 10). The slope in kg per cm equals the slope of v437 on v438.
# ==============================================================
from scipy import stats

colors = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a",
          "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
          "grid": "#e1e0d9", "axis": "#c3c2b7"}


def style_axes(ax):
    """Recessive grid and axes for the figures of the notes."""
    ax.grid(True, color=colors["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(colors["axis"])
    ax.tick_params(colors=colors["ink2"], labelsize=9)


reg = pd.DataFrame({"w": df.v437 / 10, "h": df.v438 / 10})
n = len(reg)

#%% Sample composition and the role of the data filter
raw = pd.read_stata("data/replication_final.dta", convert_categoricals=False,
                    columns=["v000", "v001", "v002", "v012", "v437", "v438"])
raw = raw.drop_duplicates(subset=["v002", "v012"])
country = raw.loc[df.index, "v000"].str[:2]   # v000 = country code + DHS phase
cluster = raw.loc[df.index, "v000"] + "_" + raw.loc[df.index, "v001"].astype(int).astype(str)  # PSU
print(f"\nn = {n} women aged {df.v012.min()}-{df.v012.max()}, "
      f"{raw.loc[df.index, 'v000'].nunique()} surveys, {country.nunique()} countries, "
      f"of which {(country == 'IA').sum()} Indian women")

# DHS codes a missing measure as 9999: what if we forgot the height filter?
valid_w = raw.loc[(raw.v437 < 1600) & raw.v438.notna()]
coded = (valid_w.v438 >= 9990).sum()
too_short = (valid_w.v438 <= 1200).sum()
slope_nofilter = smf.ols("v437 ~ v438", data=valid_w).fit().params["v438"]
print(f"Without the height filter: {len(valid_w) - n} more women "
      f"({coded} heights coded 9999, {too_short} below 120 cm); "
      f"slope = {slope_nofilter:.3f} kg per cm instead of {modelOLS.params['v438']:.3f}")

#%% Regression as a conditional mean: OLS line and binned means
r = reg[["h", "w"]].corr().iloc[0, 1]
ols_cl = smf.ols("w ~ h", data=reg).fit()                    # classical SEs
ols = smf.ols("w ~ h", data=reg).fit(cov_type="HC1")         # robust SEs
b0, b1 = ols.params["Intercept"], ols.params["h"]

reg["bin"] = pd.qcut(reg.h, 20, labels=False)
binned = reg.groupby("bin").agg(h=("h", "mean"), w=("w", "mean"))
binned["gap"] = binned.w - (b0 + b1 * binned.h)
print(f"\nBinned means (20 groups): largest gap to the OLS line = "
      f"{binned.gap.abs().max():.2f} kg")

#%% Algebraic properties of OLS (they hold in every sample)
e = ols.resid
print("\nAlgebra of OLS:")
print(f"  sum of residuals = {e.sum():.2e}, sum of h * residuals = {(reg.h * e).sum():.2e}")
print(f"  line through the means: b0 + b1 * hbar = {b0 + b1 * reg.h.mean():.3f} "
      f"= wbar = {reg.w.mean():.3f}")
print(f"  r = {r:.4f}, r^2 = {r**2:.4f}, R2 = {ols.rsquared:.4f}, "
      f"standardized slope b1 * s_h / s_w = {b1 * reg.h.std() / reg.w.std():.4f}")
print(f"  s_w = {reg.w.std():.2f} kg, s_h = {reg.h.std():.2f} cm, "
      f"hbar = {reg.h.mean():.2f} cm, wbar = {reg.w.mean():.2f} kg, "
      f"heights from {reg.h.min():.1f} to {reg.h.max():.1f} cm")

#%% Table: with intercept, height centred, without intercept
reg["h_c"] = reg.h - reg.h.mean()
cen_cl = smf.ols("w ~ h_c", data=reg).fit()
cen = smf.ols("w ~ h_c", data=reg).fit(cov_type="HC1")
noi_cl = smf.ols("w ~ h - 1", data=reg).fit()
noi = smf.ols("w ~ h - 1", data=reg).fit(cov_type="HC1")
tss = ((reg.w - reg.w.mean()) ** 2).sum()
print("\nTable (HC1 = heteroskedasticity-robust standard errors):")
print(f"{'':28s}{'(1) intercept':>15s}{'(2) h centred':>15s}{'(3) no intercept':>18s}")
print(f"{'slope, kg per cm':28s}{b1:15.3f}{cen.params['h_c']:15.3f}{noi.params['h']:18.3f}")
print(f"{'  classical SE':28s}{ols_cl.bse['h']:15.4f}{cen_cl.bse['h_c']:15.4f}{noi_cl.bse['h']:18.4f}")
print(f"{'  HC1 SE':28s}{ols.bse['h']:15.4f}{cen.bse['h_c']:15.4f}{noi.bse['h']:18.4f}")
print(f"{'intercept, kg':28s}{b0:15.2f}{cen.params['Intercept']:15.2f}{'--':>18s}")
print(f"{'  HC1 SE':28s}{ols.bse['Intercept']:15.2f}{cen.bse['Intercept']:15.2f}{'--':>18s}")
print(f"{'R2 (centred)':28s}{ols.rsquared:15.3f}{cen.rsquared:15.3f}"
      f"{1 - (noi.resid ** 2).sum() / tss:18.3f}")
print(f"{'R2 reported by statsmodels':28s}{ols.rsquared:15.3f}{cen.rsquared:15.3f}"
      f"{noi.rsquared:18.3f}")
print(f"{'SER s, kg':28s}{np.sqrt(ols.scale):15.2f}{np.sqrt(cen.scale):15.2f}"
      f"{np.sqrt(noi.scale):18.2f}")
print(f"{'mean residual, kg':28s}{ols.resid.mean():15.2f}{cen.resid.mean():15.2f}"
      f"{noi.resid.mean():18.2f}")
print(f"{'n':28s}{n:15d}{n:15d}{n:18d}")

#%% Reverse regression and regression to the mean
rev = smf.ols("h ~ w", data=reg).fit()
g1 = rev.params["w"]
print(f"\nReverse regression: {g1:.3f} cm per kg; 1/{g1:.3f} = {1 / g1:.2f} kg per cm "
      f"vs b1 = {b1:.3f}; b1 * g1 = {b1 * g1:.4f} = r^2")
h2 = reg.h.mean() + 2 * reg.h.std()
w2 = reg.w.mean() + 2 * reg.w.std()
print(f"A woman 2 SD taller ({h2:.1f} cm) is predicted {b0 + b1 * h2:.1f} kg, "
      f"i.e. {2 * r:.2f} SD above mean weight")
print(f"A woman 2 SD heavier ({w2:.1f} kg) is predicted "
      f"{rev.params['Intercept'] + g1 * w2:.1f} cm, i.e. {2 * r:.2f} SD above mean height")

#%% Inference on the slope
tcrit = stats.t.ppf(0.975, n - 2)
print(f"\nInference: t critical (5%, two-sided, n-2 df) = {tcrit:.3f}")
for ct in ["HC0", "HC1", "HC2", "HC3"]:
    print(f"  {ct} SE = {smf.ols('w ~ h', data=reg).fit(cov_type=ct).bse['h']:.4f}")
print(f"  classical SE = {ols_cl.bse['h']:.4f}, t = {ols_cl.tvalues['h']:.1f}")
print(f"  robust / classical SE = {ols.bse['h'] / ols_cl.bse['h']:.3f}")
print(f"  HC1 SE = {ols.bse['h']:.4f}, t = {ols.tvalues['h']:.1f}, "
      f"95% CI = [{ols.conf_int().loc['h', 0]:.3f}, {ols.conf_int().loc['h', 1]:.3f}]")
# The DHS samples households in clusters (primary sampling units, v001 within a survey)
ols_clu = smf.ols("w ~ h", data=reg).fit(cov_type="cluster",
                                         cov_kwds={"groups": pd.factorize(cluster)[0]})
print(f"  cluster-robust SE ({cluster.nunique()} clusters) = {ols_clu.bse['h']:.4f}, "
      f"95% CI = [{ols_clu.conf_int().loc['h', 0]:.3f}, {ols_clu.conf_int().loc['h', 1]:.3f}]")

# Mean response versus individual prediction at 160 cm
pred = ols_cl.get_prediction(pd.DataFrame({"h": [160]})).summary_frame(alpha=0.05)
print(f"At 160 cm: predicted weight {pred['mean'].iloc[0]:.2f} kg; "
      f"95% CI of the mean [{pred['mean_ci_lower'].iloc[0]:.2f}, "
      f"{pred['mean_ci_upper'].iloc[0]:.2f}]; 95% prediction interval for one woman "
      f"[{pred['obs_ci_lower'].iloc[0]:.1f}, {pred['obs_ci_upper'].iloc[0]:.1f}]")

#%% Sampling distribution of the OLS slope: our 15,494 women as the population
# Draws with replacement are i.i.d. from the population, as in assumption (A2).
rng = np.random.default_rng(2026)
x_pop, y_pop = reg.h.to_numpy(), reg.w.to_numpy()
u_pop = y_pop - b0 - b1 * x_pop                          # population BLP errors
var_x = x_pop.var()
sizes, R = [50, 200, 1000], 5000
draws = {}
print(f"\nMonte Carlo, {R} samples per size; population slope = {b1:.3f}")
for size in sizes:
    slopes = np.empty(R)
    for j in range(R):
        idx = rng.integers(0, n, size)
        xs, ys = x_pop[idx], y_pop[idx]
        xc = xs - xs.mean()
        slopes[j] = (xc * (ys - ys.mean())).sum() / (xc ** 2).sum()
    draws[size] = slopes
    sd_robust = np.sqrt(np.mean((x_pop - x_pop.mean()) ** 2 * u_pop ** 2) / (size * var_x ** 2))
    sd_classical = np.sqrt(np.mean(u_pop ** 2) / (size * var_x))
    print(f"  n = {size:5d}: mean = {slopes.mean():.3f} (MC s.e. {slopes.std() / np.sqrt(R):.4f}), "
          f"sd = {slopes.std():.3f}, robust formula = {sd_robust:.3f}, "
          f"homoskedastic formula = {sd_classical:.3f}")

#%% Diagnostics: heteroskedasticity, leverage and influence, normality
logm = smf.ols("np.log(w) ~ h", data=reg).fit()
reg["q"] = pd.qcut(reg.h, 5, labels=False)
reg["e"], reg["e_log"] = ols.resid, logm.resid
quint = reg.groupby("q").agg(h=("h", "mean"), sd_e=("e", "std"), sd_e_log=("e_log", "std"))
print("\nResidual standard deviation by height quintile:")
print(quint.round(3))
print(f"  ratio top/bottom: levels {quint.sd_e.iloc[-1] / quint.sd_e.iloc[0]:.2f}, "
      f"logs {quint.sd_e_log.iloc[-1] / quint.sd_e_log.iloc[0]:.2f}")
bp = het_breuschpagan(ols.resid, ols_cl.model.exog)        # LM = n R2 of e^2 on (1, h)
wh = het_white(ols.resid, ols_cl.model.exog)               # adds h^2
bp_log = het_breuschpagan(logm.resid, logm.model.exog)
z_h = (reg.h - reg.h.mean()) / reg.h.std()
tails = z_h.abs() > 3                            # women more than 3 SD from mean height
print(f"  {tails.sum()} women more than 3 SD from mean height: residual SD = "
      f"{reg.e[tails].std():.2f} kg vs {reg.e[~tails].std():.2f} kg for the others; "
      f"(robust / classical variance) = {(ols.bse['h'] / ols_cl.bse['h']) ** 2:.3f}")
print(f"  Breusch-Pagan LM = {bp[0]:.1f} (p = {bp[1]:.1e}); White LM = {wh[0]:.1f} "
      f"(p = {wh[1]:.1e}); Breusch-Pagan, log weight: LM = {bp_log[0]:.2f} (p = {bp_log[1]:.3f})")

infl = ols_cl.get_influence()
hat = infl.hat_matrix_diag
dfbeta = infl.dfbeta[:, 1]                       # change in slope when i is deleted
print(f"\nLeverage: mean h_ii = {hat.mean():.6f} = 2/n = {2 / n:.6f}; "
      f"max h_ii = {hat.max():.4f} (height {reg.h.iloc[hat.argmax()]:.1f} cm), "
      f"{hat.max() / hat.mean():.0f} times the mean")
print(f"Influence: largest change in the slope when one woman is deleted = "
      f"{np.abs(dfbeta).max():.4f} kg per cm ({100 * np.abs(dfbeta).max() / b1:.1f}% of b1)")
print(f"Residuals: skewness = {stats.skew(e):.2f}, kurtosis = {stats.kurtosis(e, fisher=False):.1f}, "
      f"Jarque-Bera = {stats.jarque_bera(e).statistic:.0f}")

#%% Figures of the notes
if ploton:
    # Figure 1: OLS with and without intercept (left), binned conditional means (right)
    grid_h = np.linspace(reg.h.min(), reg.h.max(), 200)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), dpi=150)
    ax = axes[0]
    ax.scatter(reg.h, reg.w, s=4, color=colors["muted"], alpha=0.25,
               edgecolors="none", rasterized=True, label="Women")
    ax.plot(grid_h, b0 + b1 * grid_h, color=colors["orange"], linewidth=2,
            label="OLS with intercept")
    ax.plot(grid_h, noi.params["h"] * grid_h, color=colors["aqua"], linewidth=2,
            linestyle="--", label="OLS without intercept")
    ax.set_title("All women", fontsize=10, color=colors["ink"])
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax = axes[1]
    ax.plot(grid_h, b0 + b1 * grid_h, color=colors["orange"], linewidth=2,
            label="OLS line")
    ax.plot(binned.h, binned.w, linestyle="none", marker="o", markersize=6,
            color=colors["blue"], markeredgecolor="white", markeredgewidth=0.8,
            label="Mean weight, 20 height groups")
    ax.set_xlim(binned.h.min() - 3, binned.h.max() + 3)
    ax.set_ylim(binned.w.min() - 4, binned.w.max() + 4)
    ax.set_title("Conditional means", fontsize=10, color=colors["ink"])
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    for ax in axes:
        style_axes(ax)
        ax.set_xlabel("Height (cm)", fontsize=9, color=colors["ink2"])
        ax.set_ylabel("Weight (kg)", fontsize=9, color=colors["ink2"])
    fig.tight_layout()
    fig.savefig("fig/weight_height_binned_means.pdf", bbox_inches="tight")
    plt.show()

    # Figure 2: sampling distribution of the slope for three sample sizes
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.1), dpi=150, sharex=True)
    grid_b = np.linspace(-0.3, 1.8, 400)
    edges = np.arange(-0.3, 1.8 + 1e-9, 0.025)          # same bins in every panel
    for ax, size in zip(axes, sizes):
        sd_robust = np.sqrt(np.mean((x_pop - x_pop.mean()) ** 2 * u_pop ** 2)
                            / (size * var_x ** 2))
        ax.hist(draws[size], bins=edges, density=True, color=colors["blue"],
                edgecolor="white", linewidth=0.3, label=f"{R:,} OLS slopes")
        ax.plot(grid_b, stats.norm.pdf(grid_b, b1, sd_robust), color=colors["orange"],
                linewidth=1.5, label="Normal approximation, robust variance")
        ax.axvline(b1, color=colors["ink"], linewidth=0.8, label="Population slope")
        ax.text(0.97, 0.90, f"s.d. = {draws[size].std():.3f}", transform=ax.transAxes,
                ha="right", fontsize=8, color=colors["ink2"])
        ax.set_title(f"n = {size:,}", fontsize=10, color=colors["ink"])
        ax.set_xlabel("Slope (kg per cm)", fontsize=9, color=colors["ink2"])
        ax.set_yticks([])
        style_axes(ax)
        ax.spines["left"].set_visible(False)
    axes[0].set_xlim(-0.3, 1.8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig("fig/weight_height_OLS_sampling.pdf", bbox_inches="tight")
    plt.show()


#%% Log-linear regression

# take the log of weights
df['logv437'] = np.log(df.v437)

if ploton:
    ax = df.plot.scatter(x= "v438",y="logv437")
    ax.set_xlabel("height")
    ax.set_ylabel("weight")
    fig = ax.get_figure()
    fig.savefig('fig/Indian_height_logweight_scatter.pdf')

modellogOLS = smf.ols('logv437 ~  v438',data = df).fit()
modellogLaTeX = modellogOLS.summary().as_latex()
print(modellogOLS.summary())

# adding the quadratic term
modellog_quad = smf.ols("logv437 ~ v438 + I(v438**2)", data=df).fit()

print(modellog_quad.summary())

#%% ------------------------------------------------------------
# (A5) Homoskedasticity: Breusch–Pagan and White tests
# ------------------------------------------------------------
# H0 (Breusch–Pagan and White): Var(e_i | X) = sigma^2  (homoskedasticity)
# H1: Var(e_i | X) depends on X (heteroskedasticity)
# BP requires exog matrix; include constant
residlog = modellogOLS.resid
exoglog = modellogOLS.model.exog  # already includes intercept
bp_lm, bp_lmpval, bp_f, bp_fpval = het_breuschpagan(residlog, exoglog)
print("\n(A5) Homoskedasticity tests:")
print("  Breusch–Pagan:")
print(f"    LM stat = {bp_lm:.4f}, p-value = {bp_lmpval:.4g}")
print(f"    F  stat = {bp_f:.4f}, p-value = {bp_fpval:.4g}")

#%% Log-(1 + y) linear regression

# take the log of weights
df['logoneplusv437'] = np.log(1+df.v437)

if ploton:
    ax = df.plot.scatter(x= "v438",y="logoneplusv437")
    ax.set_xlabel("height")
    ax.set_ylabel("weight")
    fig = ax.get_figure()
    fig.savefig('fig/Indian_height_logoneplusweight_scatter.pdf')

modellogoneplusOLS = smf.ols('logoneplusv437 ~  v438',data = df).fit()
modellogLaTeX = modellogoneplusOLS.summary().as_latex()
print(modellogoneplusOLS.summary())

#%% Poisson regression
# Given the nature of the data, we are treating weight as count-like data. 
# While this might not be the ideal model, it gives a Poisson regression coherent with the log-linear model

# Using Poisson regression
poisson_model = sm.GLM(df.v437, sm.add_constant(df.v438), family=sm.families.Poisson()).fit()
poisson_model_summary = poisson_model.summary()
print(poisson_model_summary)

# The link function in the Poisson model is the log function, making this coherent with the log-linear regression


#%% RANDOM FOREST REGRESSION


# Try to get a comparable summary() outcome
def sensitivity_approximation_rf(model, X):
    """Compute the average marginal effect of X on predictions for Random Forest."""
    # Compute predictions with original X
    original_preds = model.predict(X)
    
    # Increment X by a small value
    delta_X = X + 1
    incremented_preds = model.predict(delta_X)
    
    # Compute average change in prediction per unit increase in X
    return np.mean(incremented_preds - original_preds)

def random_forest_summary(model, X, y):
    n_trees = model.n_estimators
    feature_importances = model.feature_importances_
    
    sensitivity_coef = sensitivity_approximation_rf(model, X)
    
    print("Number of Trees:", n_trees)
    print("Feature Importances:")
    print(f"\tMean: {feature_importances.mean()}")
    print(f"\tStandard Deviation: {feature_importances.std()}")
    if model.oob_score:
        print("Out-of-Bag Score:", model.oob_score_)
    print("R^2 Score:", model.score(X, y))
    rmse = np.sqrt(mean_squared_error(y, model.predict(X)))
    print("Root Mean Squared Error:", rmse)
    print(f"Average Sensitivity Coefficient for v438: {sensitivity_coef}")
    print("-------------------------------------------------------")

clf = RandomForestRegressor(oob_score=True)  # Activate out-of-bag score
clfit = clf.fit(X=df.v438.values.reshape(-1, 1), y=df.logv437.values.ravel())

clfrestrict = RandomForestRegressor(n_estimators=10, max_depth=4, bootstrap=False)
clfitrestrict = clfrestrict.fit(X=df.v438.values.reshape(-1, 1), y=df.logv437.values.ravel())

print("Summary for Random Forest Regressor:")
random_forest_summary(clfit, df.v438.values.reshape(-1, 1), df.logv437.values.ravel())

print("Summary for Restricted Random Forest Regressor:")
random_forest_summary(clfitrestrict, df.v438.values.reshape(-1, 1), df.logv437.values.ravel())


#%% Predict with RANDOM FOREST REGRESSION


dfpred = pd.DataFrame(index=range(np.int32(df.v438.min()),np.int32(df.v438.max())),columns=['OLS','RF','RF restricted'])

dfpred.OLS = modellogOLS.params[0] + modellogOLS.params[1] * dfpred.index

dfpred.RF = clfit.predict(np.array(dfpred.index).reshape(-1, 1))

dfpred['RF restricted'] = clfitrestrict.predict(np.array(dfpred.index).reshape(-1, 1))

dfpred = pd.DataFrame(index=range(np.int32(df.v438.min()),np.int32(df.v438.max())),columns=['OLS','RF','RF restricted'])

dfpred.OLS = modellogOLS.params[0] + modellogOLS.params[1] * dfpred.index

dfpred.RF = clfit.predict(np.array(dfpred.index).reshape(-1, 1))

dfpred['RF restricted'] = clfitrestrict.predict(np.array(dfpred.index).reshape(-1, 1))

if ploton:
    ax = dfpred.plot()
    ax.set_xlabel("height")
    ax.set_ylabel("predicted log weight")
    fig = ax.get_figure()
    #fig.savefig('fig/Indian_OLS_vs_RF.pdf')
    

