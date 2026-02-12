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
from scipy.stats import chi2_contingency, spearmanr
from statsmodels.stats.diagnostic import (
    het_breuschpagan,
    het_white,
    linear_reset,
)
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.outliers_influence import OLSInfluence

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

#%% Are the rv statiscially independent?

# the null hypothesis is that there is no significant relationship between weight and height
chi2_contingency(df.loc[:,['v437','v438']]) # p-value = 0.0, risk to wrongly reject H_0 so low we can safely reject H_0

# ---------------------------------------------------------------------
# Basic association checks (use correlation / Spearman rather than chi2)
# ---------------------------------------------------------------------
pearson_r = df[["v437", "v438"]].corr().iloc[0, 1]
spearman_r, spearman_p = spearmanr(df["v437"], df["v438"], nan_policy="omit")

print(f"Pearson corr(weight,height):  {pearson_r:.4f}")
print(f"Spearman corr(weight,height): {spearman_r:.4f} (p={spearman_p:.2e})")


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
np.abs(betacalcule - modelOLS_no_intercept.params[0]) < 10**(-5)

# manual computation of the beta for the model with a constant
beta1calcule = (df.v438 * (df.v437 - df.v437.mean())).sum() / (df.v438 * (df.v438-df.v438.mean())).sum()

# make sure that our manual computation is close enough to the estimation we get in python
np.abs(beta1calcule - modelOLS.params[1]) < 10**(-5)

beta0calcule = df.v437.mean() - beta1calcule * df.v438.mean()

# manual computation of the beta for the model with a constant
(beta0calcule - modelOLS.params[0]) < 10**(-5)

# Visual of both regressions
# we tae the constant and the estimated beta
alpha = modelOLS.params[0]
beta = modelOLS.params[1]
beta2 = modelOLS_no_intercept.params[0]


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
    

