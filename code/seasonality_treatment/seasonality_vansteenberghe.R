##############################################################################
# seasonality_vansteenberghe.R
#
# Quantitative Methods in Finance (QMF) — Lecture Notes Companion Code
# Section: "R: Treatment of seasonality in our time series" (Sec. \ref{sec:seasonality})
#
# Author: Eric Vansteenberghe
# Affiliation: Banque de France & Université Paris 1 Panthéon-Sorbonne
# Year: 2026
#
# Purpose
# ---------------------------
# End-to-end, pedagogical workflow for diagnosing and treating non-stationarity
# in monthly time series with (i) increasing variability, (ii) trend, and (iii)
# seasonality. The code emphasizes that seasonality and integration should be
# handled jointly via SARIMA (maximum likelihood estimation), rather than by
# sequential “ad hoc” detrending/seasonal differencing when the goal is a
# coherent econometric specification.
#
# Main components
# ---------------
# (1) Bourbonnais & Terraza (2010) exercise: Box–Cox / log transformation,
#     seasonal differencing (lag 12), and ADF testing under seasonality/trend.
# (2) Coffee prices (Brazil): Buys–Ballot table + two-way ANOVA-style variance
#     decomposition (month vs year effects) and monthly dummy regression test.
# (3) Australian rainfall: seasonality diagnosis, Box–Cox, seasonal differencing,
#     ARMA identification, SARIMA estimation, residual diagnostics, forecasting,
#     and STL / classical decomposition; HP filter illustration on trend.
# (4) Melbourne rainfall: logistic regression for rainy-day occurrence using
#     lagged rainfall and trigonometric seasonal terms; out-of-sample validation.
# (5) Hare dataset (TSA): Box–Cox transformation, ARIMA fitting, residual checks,
#     and back-transformation for forecasts.
#
# Data inputs (relative to setwd("/Users/skimeur/Mon Drive/QMF"))
# --------------------------------------------------------------
# - data/c1ex4.xls
# - data/coffeeprices.xls
# - data/australian_rainfall_monthly.csv
# - data/MelbourneRainfall.xls
# Outputs
# -------
# - fig/rainfall_australia.pdf
# - fig/rain_melbourne.pdf
#
# References
# ----------
# - Pindyck, R. S., & Rubinfeld, D. L. (1998). Econometric Models and Economic Forecasts.
# - Bourbonnais, R., & Terraza, M. (2010). Analyse des séries temporelles.
# - Vansteenberghe, E. (2026). Quantitative methods in finance. arXiv:2601.12896.
#
# Notes
# -----
# - Requires several CRAN packages (TSA, forecast, tseries, mFilter, caret, etc.).
# - X-13ARIMA-SEATS is mentioned in the lecture notes; this script focuses on
#   SARIMA and decomposition-based approaches implemented directly in R.
##############################################################################

closeAllConnections()
rm(list=ls())

setwd("/Users/skimeur/Mon Drive/QMF")
library(TSA)
library(xts)
library(readxl)
library(reshape2)
library(xtable)
library(mFilter)
library(caret)
library(forecast)
library(gdata) # to import xls files
library(tseries) # for adf.test

##############################################################################
# Exercise: data from the book "Analyse des séries temporelles" de Bourbonnais
##############################################################################
# data downloadable from here: http://regisbourbonnais.dauphine.fr/fr/publications/analyse-des-series-temporelles.html
# Read the data from the specified sheet
df <- read_excel("data/c1ex4.xls", sheet = 1)
# Set the first column as row names
row.names(df) <- df[[1]]
# Remove the first column from the data
df <- df[-1]
adf_test_result <- adf.test(df[[1]])
print(adf_test_result) # H0: there is a unit root
# can we trust this result? two problems, trend, seasonality and increasing variability


df <- ts(df,start = c(1982,1),freq = 12)
plot(df)


# increasing variability => Box-Cox transformation of the data, let's start with a log
plot(log(df))
adf.test(log(df))

lambda <- BoxCox.lambda(df, method=c("guerrero"), lower=-5, upper=5)
df.bc <- (as.data.frame(df)**lambda-1)/lambda
df.bc <- ts(df.bc,start = c(1982,1),freq = 12)

plot(df.bc)

# monthly seasonality suspected: 12 months lag in the diff
dfD <- diff(df.bc, lag=12)

plot(dfD)
# we took care of seasonality, trend and increasing variability
adf.test(dfD)


##################################
# we import coffee price in Brazil
##################################
df <- read_excel("data/coffeeprices.xls")
colin <- as.list(t(df[31,2:13]))
df <- df[32:51,2:13]
# convert to numeric
df <- as.data.frame(sapply(df, as.numeric))
colnames(df) <- colin
row.names(df) <- 1982:2001
# exercise: find a "nicer" way to select a country a have a vector of coffee price from the data frame

# put the df into a time series
dfts <- as.vector(t(df))
dfts <- ts(dfts,start = c(1982,1),freq = 12)

plot(dfts)


adf.test(dfts) # H0: there is a unit root
# series seems non stationary

# search for monthly bias
colMeans((df))
meancol <- apply(df,2,mean)
max(meancol)-min(meancol)
sdcol <- apply(df,2,sd)


x.. <- mean(colMeans(df))
xi. <- rowMeans(df)
x.j <- colMeans(df)

Sp <- nrow(df) * sum((x.j - x..)**2)
Vp <- Sp/(ncol(df)-1)

Sa <- ncol(df) * sum((xi. - x..)**2)
Va <- Sa/(nrow(df)-1)

Sr <- 0
for(i in 1:nrow(df)){
  for(j in 1:ncol(df)){
    Sr <- Sr + (df[i,j]-xi.[i]-x.j[j]+x..)**2
  }
}

Vr <- Sr/((ncol(df)-1)*(nrow(df)-1))

# test of influence of the column factor (month play a role)

Fc <- Vp/Vr

Fc > qf(0.95, df1 = (ncol(df)-1), df2 = (nrow(df)-1)*(ncol(df)-1))
# we cannot reject HO and our series is not seasonal

# test of influence of the row factor (the year)

Fy <- Va/Vr
Fy > qf(0.95, df1 = (nrow(df)-1), df2 = (nrow(df)-1)*(ncol(df)-1))
# we can reject H0 and there seems to be a trend in our series

# second seasonality test
month <- season(dfts)
levels(month)

model <- lm (dfts ~ month)
summary(model)


##########################
# australian rainfall data
##########################

df <- read.csv("data/australian_rainfall_monthly.csv", sep=",", dec=".",row.names = 1)
df <- as.data.frame(df)
dfts <- as.vector(t(df))
dfts <- ts(dfts, start = c(1900,1), freq = 12)

pdf("fig/rainfall_australia.pdf",width=7,height=5)
plot(dfts)
dev.off()

adf.test(dfts)
# series seems stationary
# but can we trust this if the series is seasonal?

x.. <- mean(colMeans(df))
xi. <- rowMeans(df)
x.j <- colMeans(df)

Sp <- nrow(df) * sum((x.j - x..)**2)
Vp <- Sp/(ncol(df)-1)

Sa <- ncol(df) * sum((xi. - x..)**2)
Va <- Sa/(nrow(df)-1)

Sr <- 0
for(i in 1:nrow(df)){
  for(j in 1:ncol(df)){
    Sr <- Sr + (df[i,j]-xi.[i]-x.j[j]+x..)**2
  }
}

Vr <- Sr/((ncol(df)-1)*(nrow(df)-1))

# test of influence of the column factor (month play a role)

Fc <- Vp/Vr

Fc > qf(0.95, df1 = (ncol(df)-1), df2 = (nrow(df)-1)*(ncol(df)-1))
# we can reject HO and our series is seasonal

# test of influence of the row factor (the year)

Fy <- Va/Vr
Fy> qf(0.95, df1 = (nrow(df)-1), df2 = (nrow(df)-1)*(ncol(df)-1))
# we can reject H0 and there seems to be a trend in our series

# we check that there is a trend
year.end <- endpoints(dfts, on = "years")
dfts.year <- period.apply(as.xts(dfts), INDEX = year.end , FUN = mean)
plot(dfts.year)


# seasonality test
month <- season(dfts)
model <- lm (dfts ~ month)
summary(model)

plot(log(dfts))

# STEP 1: check increasing variablity
lambda <- BoxCox.lambda(dfts, method=c("guerrero"), lower=-5, upper=5)
df.bc <- (as.data.frame(dfts)**lambda-1)/lambda
df.bc <- ts(df.bc,start = c(1900,1),freq = 12)
plot(df.bc)

# STEP 2: trend and seasonality (first diff takes care of trend anyway)
dfD <- diff(df.bc, lag=12)
plot(dfD)

# we try to fit an ARMA process
layout(matrix(c(1,1,2,3), 2, 2, byrow = TRUE))
plot(armasubsets(y=dfts,nar=7,nma=7))
# autocorrelation function: which lags of the MA components are of interest
acf(dfts)
# partial autocorrelation function: which lags of the AR component are of interest
pacf(dfts)

# we chose an ARMA(4,1) model
shorten <- 1.04
plot(dfts[(length(dfts)/shorten):length(dfts)])
m1.df <- arima(x=dfts[(length(dfts)/shorten):length(dfts)],order = c(4,0,1))
m1.df <- Arima(dfts,order = c(4,0,1))

#runs(m1.df$residuals)
Box.test(m1.df$residuals, lag = 1, type = "Ljung-Box", fitdf = 0)
# p-value high enough that we accept the null hypothesis of no autocorrelation of error terms

qqnorm(rstandard(m1.df),col='blue')
qqline(rstandard(m1.df))

shapiro.test(residuals(m1.df))
# H0: normality of residuals
# in this case, we reject the null hypothesis and unfortunately the residuals are not normal

# we predict the rainfall over the next 5 years
futurVal <- forecast(m1.df, h=60, level=c(99.5))
plot(futurVal)


# time series decomposition
df.dec.mult <- decompose(dfts, type='multiplicative')
plot(df.dec.mult)
df.dec <- decompose(dfts, type='additive')
plot(df.dec)
# other approach
fit = stl(dfts, s.window="periodic")
plot(fit)

trend <- fit$time.series[,2]
seasonal <- fit$time.series[,1]
residual <- fit$time.series[,3]

dfts.deseasonalized <- dfts - seasonal

plot(dfts)
plot(dfts.deseasonalized)

# we try to fit an ARMA process
layout(matrix(c(1,1,2,3), 2, 2, byrow = TRUE))
plot(armasubsets(y=dfts.deseasonalized,nar=7,nma=7))

# fit an ARMA and check the model

# HP filter on the trend
lambda <- 129600
trend.hp <- hpfilter(trend, freq=lambda)
plot(trend.hp)


# Now try to fit an ARIMA taking the seasonality into account

hist(diff(diff(dfts,lag=12)),main="Histogram",xlab="difference")

acf(as.vector(diff(dfts,lag=12))) # MA(1)
pacf(as.vector(diff(dfts,lag=12))) # AR(1)

# find the parameters using the R function
auto.arima(dfts)

sarima301x210.12 <- arima(x=dfts,order=c(3,0,1),seasonal=list(order=c(2,1,0),period=12))
sarima301x210.12

shapiro.test(residuals(sarima301x210.12)) # we reject the null hypothesis of normally distributed residuals

fore = predict(sarima301x210.12, n.ahead=24)

ts.plot(dfts, fore$pred, col=1:2, xlim=c(2000,2019), ylab="Australian rainfall", main="sarima301x210.12")
lines(fore$pred+fore$se, lty="dashed", col=4)
lines(fore$pred-fore$se, lty="dashed", col=4)


####################
# Melbourne Rainfall
####################

# Exercise: code taken from http://www.forecastingbook.com/resources/data-and-code

rain.df <- read_excel("data/MelbourneRainfall.xls",sheet=1)
rain.df$Date <- as.Date(rain.df$Date, format="%Y-%m-%d")
colnames(rain.df) <- c("Date","Rainfall")

# plot the data
pdf("fig/rain_melbourne.pdf",width=7,height=5)
mdf<-melt(rain.df,id.vars="Date")
ggplot(data=mdf,aes(x=Date,y=value)) + geom_line(aes(color=variable),size=1.25)+scale_x_date("Date")+scale_y_continuous("rain")
dev.off()

rain.df$Rainy <- ifelse(rain.df$Rainfall > 0, 1, 0)
nPeriods <- length(rain.df$Rainy)
rain.df$Lag1 <- c(NA,rain.df$Rainfall[1:(nPeriods-1)])
rain.df$t <- seq(1, nPeriods, 1)
rain.df$Seasonal_sine = sin(2 * pi * rain.df$t / 365.25)
#pdf("fig/rain_melbournge_sine.pdf",width=7,height=5)
plot(rain.df$Seasonal_sine)
#dev.off()

rain.df$Seasonal_cosine = cos(2 * pi * rain.df$t / 365.25)
train.df <- rain.df[rain.df$Date <= as.Date("12/31/2009", format="%m/%d/%Y"), ]
train.df <- train.df[-1,]
valid.df <- rain.df[rain.df$Date > as.Date("12/31/2009", format="%m/%d/%Y"), ]
xvalid <- valid.df[, c(4,6,7)]

rainy.lr <- glm(Rainy ~ Lag1 + Seasonal_sine + Seasonal_cosine, data = train.df, family = "binomial")
summary(rainy.lr)
rainy.lr.pred <- predict(rainy.lr, xvalid, type = "response") 

confusionMatrix(ifelse(rainy.lr$fitted > 0.5, 1, 0), train.df$Rainy)
confusionMatrix(ifelse(rainy.lr.pred > 0.5, 1, 0), valid.df$Rainy)


###########
# hare data
###########

data(hare)
plot(hare,type="l")

adf.test(hare)

BoxCox.ar(hare)
out <- BoxCox.ar(hare)

# # we try a Box-Cox transformation
# find a range for lambda
range(out$lambda[out$loglike > max(out$loglike)-qchisq(0.95,1)/2])
lambda1 <- 0.5
hare.bc1 <- (as.data.frame(hare)**lambda1-1)/lambda1
adf.test(hare.bc1)

# Or we can use the Guerrero method
lambda <- BoxCox.lambda(hare,method=c("guerrero"),lower=-5, upper=5)
#tsa1<-BoxCox(df, lambda=lambda)
hare.bc <- (as.data.frame(hare)**lambda-1)/lambda
adf.test(hare.bc)

plot(armasubsets(y=hare,nar=7,nma=7))
acf(hare.bc)
pacf(hare.bc)

model1 <- arima(x=hare.bc1,order=c(3,0,0))
model <- arima(x=hare.bc,order=c(3,0,0))

# Autocorrelation of residuals
tsdiag(model)

hist(rstandard(model))
qqnorm(rstandard(model))
qqline(rstandard(model))

shapiro.test(residuals(model))
shapiro.test(residuals(model1))

# which model seems more appropriate?

############
# Projection
############

#function to transform back to original signal
bctransform <- function(x){y = (lambda*x+1)^(1/lambda)}

# ploting with 25 dates ahead
plot(model,n.ahead = 25, xlab='Year',ylab='Hare Abundance',transform = bctransform, col = 'blue')

