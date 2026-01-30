##############################################################################
# sarima_vansteenberghe.R
#
# Quantitative Methods in Finance (QMF) — Lecture Notes Companion Code
# Topic: SARIMA modelling, forecasting, and extensions (ARMA/ARFIMA; earnings seasonality)
#
# Author: Eric Vansteenberghe
# Affiliation: Banque de France & Université Paris 1 Panthéon-Sorbonne
# Year: 2026
#
# Purpose
# ---------------------------
# Illustrate practical identification, estimation, diagnostics, and forecasting
# with seasonal ARIMA (SARIMA) models, benchmarked against non-seasonal AR/ARMA
# alternatives, and extended to long-memory (fractionally integrated) processes.
# The script also provides applied examples on earnings time series with seasonal
# patterns (monthly and quarterly).
#
# Pedagogical sources
# -------------------
# Inspired by: Bennett & Hugen, "Financial Analytics with R" (time-series chapters),
# and standard time-series econometrics practice (Box–Jenkins; seasonal modelling).
#
# Main workflow (high level)
# --------------------------
# (1) Recruitment (rec) & Southern Oscillation Index (soi) — seasonal vs non-seasonal:
#     - Visual inspection, stationarity checks (ADF helper via urca::ur.df)
#     - AR(2) with seasonal AR terms (period 12) vs ARMA(3,1) vs AR(2) via ar.ols
#     - Residual diagnostics (ACF/PACF, tsdisplay, Ljung–Box)
#     - Forecast comparison and figure export (multi-panel PDF)
#     - auto.arima() suggestion and explicit SARIMA fit; in-sample RMSE table
#
# (2) Varve series — volatility stabilization and long memory:
#     - Variance diagnostics; log transform comparison
#     - MA(1) and ARMA(1,1) on differences of log series
#     - Grid search illustration for MA parameter intuition
#     - Fractional differencing via fracdiff; comparison of residual ACF and log-likelihood
#
# (3) Earnings time series — seasonality testing and SARIMA fitting:
#     - Shiller S&P 500 earnings (monthly): seasonal dummy regression, seasonal differencing,
#       ARIMA/SARIMA estimation and diagnostics; forecasting with back-transform
#     - Johnson & Johnson quarterly earnings (JJ): quarterly seasonality, SARIMA(·)×(·)[4]
#
# Data inputs (relative to setwd("/Users/skimeur/Mon Drive/QMF"))
# --------------------------------------------------------------
# - Built-in datasets (astsa/TSA): soi, rec, varve, JJ
# - Excel files:
#   * data/ie_data.xls   (Shiller dataset; sheet "Data")
#
# Outputs
# -------
# - fig/recruitment_SOI.pdf
# - fig/acf_logvarve.pdf
# - fig/search_theta.pdf
#
# Packages
# --------
# astsa, TSA, zoo, xts, urca, forecast, ggplot2, fracdiff, fGarch,
# reshape, gdata, readxl
#
# Notes
# -----
# - The ADF() helper is a minimal wrapper around urca::ur.df (no deterministic terms, lags=0).
#   For applied work, consider allowing drift/trend and lag selection.
# - Forecast plots show ±1 s.e. bands from predict() for comparability across models.
##############################################################################


# Clear the workspace and close all connections
closeAllConnections()
rm(list = ls())

# Helper function to plot predictions
plot_predictions <- function(data, predictions, se, title) {
  ts.plot(data, predictions, col = 1:2, xlim = c(1980, 1990), ylab = "Recruitment", main = title)
  lines(predictions + se, lty = "dashed", col = 4)
  lines(predictions - se, lty = "dashed", col = 4)
}

# Set working directory
setwd("/Users/skimeur/Mon Drive/QMF")

# Load necessary libraries
libraries <- c("astsa", "TSA", "zoo", "xts", "urca", "forecast", "ggplot2",
               "fracdiff", "fGarch", "reshape", "gdata", "readxl")
lapply(libraries, require, character.only = TRUE)

# Define custom functions

ADF <- function(x) {
  result <- ur.df(x, type = "none", lags = 0)@teststat
  if (result < qnorm(c(.01, .05, .1)/2)[3]) {
    return("The series is stationary at the 90% critical value.")
  } else {
    return("The series is not stationary at the 90% critical value.")
  }
}

# Data Visualization and Exploration
plot(soi, ylab = "", xlab = "", main = "Southern Oscillation Index")
plot(rec, ylab = "", xlab = "", main = "Recruitment")

# for both, an additive approach would seem sufficient

cat(ADF(soi), "\n")
cat(ADF(rec), "\n")

plot(armasubsets(y = rec, nar = 5, nma = 5))
acf2(rec)

# Model Estimations
# AR(2) with seasonality
rec.seas <- arima(x = rec, order = c(2, 0, 0), seasonal = list(order = c(2, 0, 0), period = 12))
fore <- predict(rec.seas, n.ahead = 48)

tsdisplay(residuals(rec.seas))
Box.test(residuals(rec.seas), lag=16, fitdf=4, type="Ljung")
# p-value high enough that we accept the null hypothesis of no autocorrelation of error terms


# ARMA(3,1)
arma31 <- arima(x = rec, order = c(3, 0, 1))
fore31 <- predict(arma31, n.ahead = 48)

tsdisplay(residuals(arma31))
Box.test(residuals(arma31), lag=16, fitdf=4, type="Ljung")
# p-value high not enough so we reject the null hypothesis of no autocorrelation of error terms


# AR(2), second estimation method
regr <- ar.ols(rec, order = 2, demean = FALSE, intercept = TRUE)
fore2 <- predict(regr, n.ahead = 48)

# Plotting results
pdf("fig/recruitment_SOI.pdf", width = 7, height = 10)  # Increase height from 5 to 10
par(mfrow = c(3, 1))
plot_predictions(rec, fore$pred, fore$se, "AR(2) with seasonality")
plot_predictions(rec, fore2$pred, fore2$se, "AR(2)")
plot_predictions(rec, fore31$pred, fore31$se, "ARMA(3,1)")
dev.off()

# Automated parameter tuning
auto.arima(rec)

# Optimal parameter values: it is a SARIMA
p <- 1; d <- 1; q <- 0
P <- 0; D <- 0; Q <- 2
s <- 12
rec.mod <- arima(x = rec, order = c(p, d, q), seasonal = list(order = c(P, D, Q), period = s))

print(rec.mod$loglik)

# ============================================================
# 1) Forecast for rec.mod (same style as others)
# ============================================================
fore_mod <- predict(rec.mod, n.ahead = 48)

# ============================================================
# 2) In-sample RMSE comparison (4 models)
#    RMSE computed from in-sample residuals (one-step-ahead errors)
# ============================================================
rmse <- function(e) sqrt(mean(e^2, na.rm = TRUE))

rmse_tbl <- data.frame(
  Model = c("AR(2) + seasonal AR(2) [12]",
            "ARMA(3,1)",
            "SARIMA(1,1,0)x(0,0,2)[12] (rec.mod)"),
  RMSE = c(
    rmse(residuals(rec.seas)),
    rmse(residuals(arma31)),
    rmse(residuals(rec.mod))
  )
)

# Pretty print in console
rmse_tbl <- rmse_tbl[order(rmse_tbl$RMSE), ]
print(rmse_tbl, row.names = FALSE)


# ============================================================
# 3) Plotting results (now 4 panels)
# ============================================================
pdf("fig/recruitment_SOI.pdf", width = 7, height = 12)  # taller for 4 panels
par(mfrow = c(4, 1), mar = c(3.5, 4, 3, 1))

plot_predictions(rec, fore$pred,     fore$se,     "AR(2) with seasonality")
plot_predictions(rec, fore2$pred,    fore2$se,    "AR(2)")
plot_predictions(rec, fore31$pred,   fore31$se,   "ARMA(3,1)")
plot_predictions(rec, fore_mod$pred, fore_mod$se, "SARIMA(1,1,0)x(0,0,2)[12] (rec.mod)")

dev.off()



########################################
# Varves - fractionally intergrated ARMA
########################################

data(varve)
plot(varve, main = "varve")
plot(log(varve),main="log(varve)")

# show the evolution of the variance over time
# compute the standard deviation on 50 obsevations
vol.varve <-NA
for (i in 25:(length(varve)-25)){
  vol.varve[i] <- sd(varve[(i-25):(i+25)])
}
plot(vol.varve,type="l")
vol.lvarve <-NA
for (i in 25:(length(varve)-25)){
  vol.lvarve[i] <- sd(log(varve[(i-25):(i+25)]))
}
plot(vol.lvarve,type="l")

# did the log transform "improve" the variability of the volatility over time?
sd(vol.varve,na.rm=TRUE)/mean(vol.varve,na.rm=TRUE)
sd(vol.lvarve,na.rm=TRUE)/mean(vol.lvarve,na.rm=TRUE)

# we plot both distributions
hist(varve,breaks=50)
hist(log(varve),breaks=50)
# which one looks closer to a normal distribution?
# exercise: propose a test

acf(varve)

pdf("fig/acf_logvarve.pdf",width=7,height=5)
acf(log(varve))
dev.off()
acf(diff(varve))
acf(diff(log(varve)))

# exercise: model one of the transformed series as an MA(1)

# fit an MA(1) on diff of log
v.ma1 <- arima(x=diff(log(varve)),order=c(0,0,1))
summary(v.ma1)

# searching for theta
theta.seq <- seq(-0.1,-0.9,-0.01)
S.theta <- NULL
for(theta in theta.seq){
  surprise <- varve[1]
  for(i in 2:length(varve)){
    surprise <- c(surprise,(varve[i] + theta*surprise[i-1]))
  }
  S.theta <- c(S.theta,sum(surprise**2))
}

pdf("fig/search_theta.pdf",width=7,height=5)
plot(theta.seq,S.theta,type="l")
dev.off()

# plot the autocorrelation of the residual of the model

# we fit an ARMA(1,1)
v.arma11 <- arima(x=diff(log(varve)),order=c(1,0,1))
summary(v.arma11)
# comment on the results

# fractional difference
v.fd <- fracdiff(log(varve)-mean(log(varve)),nar=0,nma=0,M=30)

# value of d:
v.fd$d

# we compare the autocorrelation of the residuals of the models
acf(resid(v.arma11))
acf(resid(v.fd))

# we compare the log likelyhood
v.arma11$loglik
v.fd$log.likelihood

###################
# Earnings models
#################

# we import the Robert Shiller data with S&P 500 earnings data
df.o <- read_excel("data/ie_data.xls", sheet = "Data")
colin <- as.list(t(df.o[6,]))
df.o <- df.o[7:1755,]
colnames(df.o) <- colin
df.o <- as.data.frame(df.o)
rownames(df.o) <- seq(as.Date("1871/01/01"), by = "month", length.out = 1749)
#rownames(df.o) <- seq(as.Date("1871/01/01"), by = "month", length.out = 1749)
df.o$Date <- NULL
# convert to numeric
df.o[] <- lapply(df.o, function(x) {
  if(is.character(x)) {
    as.numeric(x)
  } else if(is.factor(x)) {
    as.numeric(as.character(x))
  } else {
    x
  }
})
# remove columns with only nas
df.o <- df.o[,colSums(is.na(df.o))<nrow(df.o)]

# we transform it to a time series and extract the earnings
df <- ts(df.o$E, start = c(1871,1), freq = 12)

# we plot the first difference of the log of Earnings of S&P 500
plot(log(df))

# we take the data from 1950s to 2000s
df <- window(df,1950,2000)

# test for seasonality in the earnings
month <- season(df)
model1 <- lm (df ~ month - 1)
summary(model1)

# we plot the first difference of the log of Earnings of S&P 500
plot(diff(log(df)))

series <- diff(diff(log(df),lag=12))

ADF(series)

acf(as.vector(series),ci.type='ma') # take MA(6)
pacf(as.vector(series),ci.type='ma') # take AR(1)

model <- arima(x=log(df),order=c(1,1,3), seasonal=list(order=c(0,0,1),period=12))
model
shapiro.test(residuals(model))
tsdiag(model)
plot(model,n.ahead=200,ylad='Earnings',transform=exp)

# Automated parameter tuning
auto.arima(log(df))
# Fit the suggested model

#Earnings of Johnson and Johnson:
data(JJ)
plot(JJ,col='blue')
plot(log(JJ),ylab='log(Earnings)',type='l',col='blue')
# we went from exponential to linear trend

quarter<- season(log(JJ))
model2 <- lm (log(JJ) ~ quarter - 1)
summary(model2)
# we suspect a quarterly seasonality

plot(diff(log(JJ)),ylab='log differenced',type='l',col='blue')
plot(diff(log(JJ),lag=4),ylab='seasonal diff',type='l',col='blue')
plot(diff(diff(log(JJ),lag=4)),ylab='diff differenced',type='l',col='blue')

series<-diff(diff(log(JJ),lag=4))
ADF(series)
plot(armasubsets(y=log(JJ),nar=4,nma=4)) # but how do you treat seasonality?
acf(as.vector(series))
pacf(as.vector(series))
model<-arima(x=log(JJ),order=c(1,1,1),seasonal=list(order=c(1,1,1),period=4))
model
shapiro.test(residuals(model))

plot(model,n1=c(1975,1), n.ahead=12, ylab='Earnings',transform=exp,col='blue')
