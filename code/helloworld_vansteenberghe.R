# =============================================================================
# QMF — Quantitative Methods in Finance
# Hello World in R: a fast introduction to R for finance and economics students
# =============================================================================
#
# This script accompanies Section "First steps with R - functions, loops,
# imports and exports" (and Section "Installing R tools") of the lecture notes
# *Quantitative Methods in Finance* by Eric Vansteenberghe (arXiv:2601.12896,
# SSRN 5178205), taught at Université Paris 1 Panthéon-Sorbonne
# (Master Finance, Technology & Data).
#
# Pedagogical objectives:
# - Learn, in one session, the R basics needed before starting an empirical
#   finance or economics research project
# - Read the help, set a working directory, install and load packages
# - Manipulate vectors, matrices, lists, data frames and data.tables
# - Write functions, if-else conditions, for and while loops, sapply()/apply()
# - Handle errors with try() and tryCatch() while going through data
# - Simulate random variables reproducibly with set.seed()
# - Import share prices and macro data from csv files, compute returns, plot
#   with ggplot2, and run a first linear regression with lm()
#
# Contents (RStudio: Cmd/Ctrl + Shift + O shows this outline):
#    0. Setup: packages and QMF folder
#    1. Help
#    2. Navigating
#    3. Assignment
#    4. Basic math
#    5. Classes and casting
#    6. Entering data
#    7. Plot
#    8. Exercise 1: sampling with and without replacement
#    9. Defining a function
#   10. if - else condition
#   11. eval
#   12. For loop
#   13. sapply
#   14. While loop
#   15. fsolve: root finding with uniroot()
#   16. Arrays
#   17. Matrices
#   18. Error handling: try() and tryCatch()
#   19. Setting the displayed precision
#   20. Random number generation and set.seed()
#   21. Working with strings
#   22. Data frames: create, merge, stack, import prices
#   23. Work with data.table
#   24. Lists
#   25. Exercise 2: loading data from external files
#   26. Basic commands: sequences, names, matrices
#   27. Data frames, again: indexing and subsetting
#   28. Indexing (super important)
#   29. Finding subsets with logical conditions
#   30. String manipulation: paste() and stringr
#   31. For loops, again
#   32. apply() over rows and columns
#   33. Linear regression with lm()
#   34. Sequences and matrix algebra: t(), %*%, solve()
#   35. Using a package: temporal disaggregation with tempdisagg
#
# Data (folder data/ of the repository):
# - data/pea_price.csv, data/pea_price_2.csv: daily prices of French shares
#   identified by ISIN (2006-2016), used for returns and data-frame handling
# - data/farmer_2015.csv: quarterly US data, 1929Q4-2011Q1 (house prices,
#   S&P 500, wages, unemployment), from Farmer (2015)
# - Built-in data sets: lalonde (package Matching), painters (MASS),
#   cars (datasets), swisspharma (tempdisagg)
#
# Requirements:
# - R >= 4.1 (https://cran.r-project.org/) and, ideally, RStudio Desktop
# - Packages: Matching, plyr, ggplot2, reshape, data.table, stringr, zoo,
#   MASS, tempdisagg (the Setup block below lists the missing ones)
#
# Usage:
# - Open the script in RStudio and run it line by line (Cmd/Ctrl + Enter):
#   it is meant to be read and played with, not only executed
# - Or run it at once: source("code/helloworld_vansteenberghe.R") from the
#   QMF folder. Lines that produce an error on purpose are wrapped in try()
#
# Credits: adapted and extended from Erin Hartman and Danny Hidalgo's
# introduction to R and from Bennett and Hugen, *Financial Analytics with R*
# (Cambridge University Press, 2016).
#
# How to cite:
# - Vansteenberghe, E. (2026). Quantitative Methods in Finance. arXiv:2601.12896.
#   https://doi.org/10.48550/arXiv.2601.12896
#
# File: helloworld_vansteenberghe.R
# Repository: https://github.com/skimeur/QMF
#
# License: MIT (code)
# Year: 2023-2026
# Author: Eric Vansteenberghe
# =============================================================================

# =============================================================================
# Setup: packages and QMF folder ----
# =============================================================================

# Packages used in this script. install.packages() is needed only once per
# computer; library() is needed in every new R session.
needed_packages <- c("Matching", "plyr", "ggplot2", "reshape", "data.table",
                     "stringr", "zoo", "MASS", "tempdisagg")
missing_packages <- setdiff(needed_packages, rownames(installed.packages()))
if (length(missing_packages) > 0) {
  message("Please install the missing packages first:\n  install.packages(c(\"",
          paste(missing_packages, collapse = "\", \""), "\"))")
}

# Find the QMF folder, i.e. the first folder (starting from this script, or
# from the current working directory) that contains the data/ folder.
# It is stored with options(), so it survives rm(list = ls()) below.
find_qmf_root <- function() {
  start <- getwd()
  if (requireNamespace("rstudioapi", quietly = TRUE) &&
      rstudioapi::isAvailable()) {
    p <- tryCatch(rstudioapi::getSourceEditorContext()$path,
                  error = function(e) "")
    if (nzchar(p)) start <- dirname(p)
  }
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) == 1) start <- dirname(sub("^--file=", "", file_arg))
  d <- normalizePath(start)
  repeat {
    if (file.exists(file.path(d, "data", "farmer_2015.csv"))) return(d)
    parent <- dirname(d)
    if (parent == d) {
      warning("Folder with data/ not found: set it yourself with setwd()")
      return(normalizePath(start))
    }
    d <- parent
  }
}
options(qmf.root = find_qmf_root())
setwd(getOption("qmf.root"))
getOption("qmf.root")

# What is R?
#  - language and environment mostly used for statistical computing and graphics
#  - based on S, a statistical programming language started at Bell Labs
#  - has a basis in Fortran and Scheme
#  - extensible, capable of plugging in Fortran, C, C+, and others
#  - open source (and free!)
#  - many people contribute an abundance of useful packages     


# Some Downsides?
#  - inefficient for big computation problems 
#  - bad for big data, though getting better 

# Installation
# can be found at:
# <http://cran.r-project.org/>
#  - download the binary for your OS (or install from source, if you wish)
#  - run the installer and follow the instructions

# Try to comment your code in a comprehensive manner

# Getting to R
# <http://en.wikibooks.org/wiki/R_Programming/>

# ============================================================================
# Help ----
# ============================================================================

# Most useful tool in R, help                  
# Starts an interactive html help file
if (interactive()) help.start()  # opens the HTML help in your browser

# help function for 'mean'
help(mean)

# also help for 'mean'
?mean

# search
# suppose we want to find matrix product methods
help.search("product")

# note that now, we can find the help file
help(read.csv)

# ============================================================================
# Navigating ----
# ============================================================================

# see what working directory you are in:
getwd()

# change to another working directory: here the QMF folder found in the Setup
# block above (the folder that contains data/). On your own computer you could
# also write, e.g., setwd("C:/Users/yourname/Documents/QMF")
setwd(getOption("qmf.root"))

# display the working directory, where your file can be loaded and saved, once executed
getwd()

# list all objects in your workspace (currently should be empty)
ls()

#Use a library (if not installed, use Tools, install package)
library(Matching)

# ============================================================================
# Assignment ----
# ============================================================================

# there are two way of doing assignment, <- and =
# <- was historically the way of assignment, = can also be used and should make no difference for most of our usages
x <- 5
y = 2
assign("a",10)
x
y
a

# list all objects in your workspace
ls()

# ============================================================================
# Basic math ----
# ============================================================================
z <- x + y
z
z - 2
x * 5
y / 3

# ============================================================================
# Classes and casting ----
# ============================================================================

# A class refers to the type of an object.
# Casting refers to recasting an object to another type of class
# there are a few major classes:
# integer and numeric:
x <- 1
class(x)
#x <- as.integer(x)
class(as.integer(x))
class(x)

x <- 2.9999
class(x)
as.integer(x)
# if you prefer to round
round(x)
#x <- as.integer(x)

# character:
y <- "test"
class(y)
y <- as.numeric(y)

# there is also matrix, array, list, expression, and many more, which we will encounter later

# ============================================================================
# Entering data ----
# ============================================================================

# note many facilities for importing data of various types into R:
# read.csv [comma], read.table [tab,space], read.dta [stata], read.xpt [sas]
# scan(n = 3) waits for you to type 3 numbers in the console;
# to keep the script runnable with source() we pass them as text:
p <- scan(text = "1 2 3")

p

# matrices: [can be tricky though]
# with the keyboard: s <- t(matrix(scan(n = 6), 2)) then type 1 4 2 5 3 6
s <- t(matrix(scan(text = "1 4 2 5 3 6"), 2))

s  

# in practice, you will not want to enter manually such data but rather import from a csv file (you could create a csv file for s)

# remove particular data from the workspace
rm(s,p,x,y,z,a) 

#create a vector of elements using c(), with a "Not Available" element
x = c(1.3,1.2,1.3,NA,1.4,2)

# display x
x

# get the number of elements in x
length(x)

# ============================================================================
# Plot ----
# ============================================================================

#Plot the data
plot(x,ylab = "values")

#test which element of the vector x is "Not Available"
is.na(x)

# display the opposite of the results of the above test
!is.na(x)

# select and display elements of x
x[1]
x[4]
x[x]
y <- c(1,3,2,2)
x[y]
x

rm(list=ls())
# ============================================================================
# Exercise 1: sampling with and without replacement ----
# ============================================================================
# 1) we create a vector ranging from 1 to 100, naming it echantillon
echantillon <- 1:100
# 2) you have to create a new variable "combinaison" where you randomly draw 100 elements from echantillon, with replacement
# 3) for this exercise, you have to do it by creating a variable indexchoice where you create integer between 1 and 100 randomly
help(runif)
indexchoice <- runif(100,1,100)
# 4) then you use this indexchoice as follow:
combinaison <- echantillon[indexchoice]
combinaison2 <- echantillon[as.integer(indexchoice)]
combinaison == combinaison2
combicheck <- as.integer(indexchoice)
combinaison == combicheck
#Plot the data as histogram, playing with the "breaks"
help(hist)
hist(combinaison)
# 4) try to do it without replacement
?length
for (i in 1:2){
  print(1)
}
combinaison <- c()
echantillon <- 1:100
echantilloncheck <- echantillon
for (i in 1:100){
  longueur <- length(echantillon)
  indexi <- runif(1,1,longueur)
  combinaison <- c(combinaison,echantillon[indexi])
  echantillon <- echantillon[-indexi]
}

hist(combinaison)

?sample
# here is the way to do it with a function
# with replacement
simplechantillon <- sample(1:100, 100, replace = T)
# without replacement
simplechantillonWOR <- sample(1:100, 100, replace = F)
hist(simplechantillonWOR)

# use the result of a is NA test to extract some elements of x
x = c(1.3,1.2,1.3,NA,1.4,2)
xbis <- x[!is.na(x)]
x <- xbis
rm(xbis)


# Filter for prices above 1.3
threshold <- 1.3
z = x[x > threshold]
x > threshold

# Compute the difference of the log of x (as it they were prices)
y = diff(log(x))
y
# Rounding y
round(y,2)


# ============================================================================
# Defining a function ----
# ============================================================================

# we manually define a function that returns x to the power of y
power <- function(x,y){
  return(x^y)
}

#apply this function to our vector x
x
power(x,3)

# It is important to note here that you can use a local variable in the function definition, like x and y and it won't affect the actual value of you variable x and y outside of this function
# If you need to go around this, you can use the super-assignment: <<-
powerbis <- function(x,y){
  x <<- 4
  return(x^y)
}

x
powerbis(x,2)
x
#By using this super-assignment <<- we changed the value of x even outside of our function (this is not usually wanted)

typeof(powerbis)
typeof(x)
typeof(z)

#we define another function
sincos <-function(x,y){
  return(sin(x)+cos(y))
}

# ============================================================================
# if - else condition ----
# ============================================================================

# we first assign a value to a variable (we could use a boolean) 
sinusite = 1

# depending on the value taken by the variable sinusite,
# we assign different value to the variable str
if (sinusite == 2) {
  str = "power(2,3)"
} else {
  str = "sincos(2,3)"
}

# ============================================================================
# eval ----
# ============================================================================

#Now that we have as an output a string (str), we can evaluate the string as an R expression using parse() and then evaluate it with eval() 
help(parse)
help(eval)

eval(parse(text = str))

# we check by manually executing it
sincos(2,3)

# now we check the other possible outcome
power(2,3)

# we modify the value of sinusite
sinusite = 2
eval(parse(text=str))
# Although we changed the value of sinusite, it didn't change our result

# In order to have a dynamic evaluation we can use:
evaluons<-function(sinusite){
  ifelse(sinusite==2,
    eval(parse(text="power(2,3)")),
    eval(parse(text="sincos(2,3)")))
}

# Now it is more dynamic
sinusite=1
evaluons(sinusite)

sinusite=2
evaluons(sinusite)

# ============================================================================
# For loop ----
# ============================================================================

#We can go one step further with defining a for loop 
# to apply the function to each element of the vector sinusite
# simple for loop example
nombre = 1e8
for (i in 1:10) {
  nombre = nombre/i
}

print(nombre)

# we show that the above loop is equivalent to 
# dividing by the factorial 1 * 2 * 3 * etc.
nombre2 = 1e8
print(nombre2/factorial(10))

#Now we apply it to our case
sinusite=c(1,2)
for(i in 1:length(sinusite)){
  print(evaluons(sinusite[i]))
}

# ============================================================================
# sapply ----
# ============================================================================
#We can also simply use the function sapply() to apply our function to each element of our vector sinusite
sapply(sinusite,evaluons)

# we redefine x as a range of integer from 1 to 20
x <- (1:20)

# we manually define the cube function
cube <- function(x){return(x^3)}

# and apply the cube function to each element of x
sapply(x,cube)

# clearing our workspace
rm(list = ls())

# we want to search for (sin(x) == cos(x) in the vicinity of 3.5
# we first manually define the sine and cosine functions
f<-function(x){
  return(sin(x))
}

g<-function(x){
  return(cos(x))
}

x=c(0,0.1,0.2,1)
f(x)
g(x)

f(x)==g(x)

# plot f: x -> f(x)
# and superpose sine on top of cosine
seq(1,20,by=1) == 1:20
z <- seq(0,10,by = pi/10)
plot(z,f(z),"l")
lines(z,g(z))

# ============================================================================
# While loop ----
# ============================================================================

#we want to search in the vector [3.5,4]
y=0
difference<-abs(f(y)-g(y))
tolerance <- 0.001
steps <- 0.0001

while(difference > tolerance){
  difference<-abs(f(y)-g(y))
  y=y+steps
}
#we found an y for which sin(y) approx cos(y)
f(y)
g(y)
y

# in fact, we know it is the point of x axis where x = pi / 4
cos(pi/4)
sin(pi/4)

# we compare both points
y - pi/4

?runif
?rnorm

# ============================================================================
# fsolve: root finding with uniroot() ----
# ============================================================================

#finding cos(x)-sin(x) = 0 varying x around 0 is what the uniroot function does for us
h<-function(x){
 return(cos(x)-sin(x))
}

# Find the root using uniroot
result <- uniroot(h, interval=c(-1, 1))
print(result$root)  # This prints the value of x where h(x) is zero


result$root - pi/4

# clearing our workspace
rm(list = ls())

# ============================================================================
# Arrays ----
# ============================================================================

#We can define two vectors and bind them into an array

#Define a vector of 1 and another as a sequence
ones <- rep(1,5)
tows <- seq(1, 100, by=20)

length(ones) == length(tows)

Array=cbind(ones,tows)
Array

# Accessing an element in the array Array[row,column]
# The indexing of row and column start at 1 (when it start at 0 with Python!)
Array[2,2]
#removing the second row
Array[-2,]
# subset
Array[1:3,]

# ============================================================================
# Matrices ----
# ============================================================================

mat2by4 <- matrix(1:8, nrow=2,ncol=4)
mat2by4

# Square of this matrix
# first trial 
try(squarematfalse <- mat2by4 * t(mat2by4)) # gives you an error, why? (non-conformable dimensions)
squarematfalse <- mat2by4 * mat2by4
# * is an element-wise multiplication, so you get each element of the matrix square

# To multiply two matrices, use %*%
squaremat <- mat2by4 %*% t(mat2by4)
squaremat
#recall that 1*2+3*4+5*6+7*8 = 100

# for further matrix algebra we recommend: https://www.statmethods.net/advstats/matrix.html

# ============================================================================
# Error handling: try() and tryCatch() ----
# ============================================================================

?tryCatch

#Taken from http://www.win-vector.com/blog/2012/10/error-handling-in-r/

# we want to apply the logarithm function to a list of inputs
inputs = list(1, 2, 4, -5, 'oops', 0, 10)
# (try() only lets the script continue; inside, the loop stops at 'oops')
try(for (element in inputs) {
   print(paste("log of", element, "=", log(element)))
})
# we got an error message that stopped the loop before the end of the list

# we can use try to avoid this and go through the whole list
for (input in inputs) {
  try(print(paste("log of", input, "=", log(input))))
}

#To chose what to return in case of error, you can use tryCatch
for (input in inputs) {
tryCatch(print(paste("log of", input, "=", log(input))),
warning = function(w) {print(paste("negative argument", input));log(-input)},
error = function(e) {print(paste("non-numeric argument", input));NaN})
}

# clearing our workspace
rm(list = ls())

# ============================================================================
# Setting the displayed precision ----
# ============================================================================

# you can chose how many digits are shown
# reminder: using 64-bit doubles your target accuracy is around 16 decimal digits
options(digits = 10)
pi

# ============================================================================
# Random number generation and set.seed() ----
# ============================================================================

# We want to plot the density of the normal distribution using rnorm
?rnorm
plot(density(rnorm(50,0,1)))

# Now with more observations we are getting closer to the theoretical distribution
plot(density(rnorm(1e6,0,1)))

# We can get random samples
set.seed(1)
sample(10)
set.seed(1)
sample(10)



#If we want to make our random sample reproducible, we can use set.seed()
set.seed(1)
x <- rbinom(10,10,0.1)
y <- rbinom(10,10,0.1)
set.seed(1)
z <- rbinom(10,10,0.1)

# x and z are random but "fixed"
x == y
x == z

# rm(.Random.seed, envir=.GlobalEnv)
# to reset seed

# an interesting read on how random numbers can be generated
# and the concept of seed: https://medium.com/delta-force/how-computers-make-random-numbers-51e8938d9d53


# ============================================================================
# Working with strings ----
# ============================================================================

# Create a list of strings
tickers <- c("RNO","SG","BNP")

# Find in which position SG is located
match("SG",tickers)

# Concatenate strings
liste <- paste("RNO","SG","BNP",sep = ",")
print(liste)

# Get a substring
substr(liste,5,6)

# ============================================================================
# Data frames: create, merge, stack, import prices ----
# ============================================================================

# Data Frames is a sequence of rows where the columns are heterogeneously typed

L3 <- LETTERS[1:3]
fac <- sample(L3 , 10 , replace = TRUE)
df <- data.frame(x = 1,y = 1:10,fac=fac)
df
df$fac

# Working on the DataFrame column names
names(df)
# change the 'fac' column name to 'factor'
names(df) <- c(names(df)[1:2],"factor")

#We can write this DataFrame to a csv file
write.csv(df, file = file.path(tempdir(), "df.csv"), row.names = FALSE)  # tempdir(): a temporary folder
# when opening this file, with LibreOffice for example, 
# make sure that you indicate that the text delimiter is "

# Accessing elements of a DataFrame
df[1:3,2]
# Accessing a column (as if it was a list, see below)
df[[3]]
df[,3]

# Navigating through a DataFrame
df <- data.frame(x=1:4,y=5:8,z=9:12,a=13:16)
# numbers of rows and columns
print(paste("df has",dim(df)[1],"rows and",dim(df)[2],"columns"))
# equivalently
print(paste("df has",nrow(df),"rows and",ncol(df),"columns"))
df
# if we want to do an item by item multiplication
df^2
# if we want to square it as a matrix
data.matrix(df) %*% t(data.matrix(df))

# if we want to loop manually over the data frame
for (i in 1:dim(df)[1]) {
  for (j in 1:dim(df)[2]) {
    df[i,j] <- df[i,j]^2
  }
}
df

# merging (horizontal) and rbinding (vertical) two data frames
?merge
# we take 4 data frames
df1 <- data.frame(A = 1,B=1:10,C=sample(LETTERS[1:3] , 10 , replace=TRUE))
df2 <- data.frame(B=1:10,D=11:20)
df3 <- data.frame(A=1000,B=1001:1010,C="bob")
df4 <- data.frame(B=1:9,C="alors?")

# we can merge both df1 and df2 keeping B
df.merged <- merge(df1,df2,by="B")

# if both data frame don't have the same number of rows
df.merged.bis <- merge(df1,df4, by = "B")
# if you want to keep all rows and add NAs
df.merged.bis.all <- merge(df1,df4, by = "B",all.x = TRUE)

# Note that in df.merged.bis.all, we now have an NA
df.merged.bis.all[10,4]
# use a function to confirm that this is an NA
is.na(df.merged.bis.all[10,4])
# any other cell in this data frame is not an NA, you'd get a False
is.na(df.merged.bis.all[2,2])
# find the columns in which you have an NA
# remember that True is 1 and False is 0
colSums(is.na(df.merged.bis.all))
# we find that there is on NA in column C.y

# notice that there is the negation operator !()
# "is there no NA in the column" is the answer to the question:
!(colSums(is.na(df.merged.bis.all)))

# If you want to keep rows that have no NA
# R will keep only column with True as an answer to the previous question
dffull <- df.merged.bis.all[,!(colSums(is.na(df.merged.bis.all)))]

# we can stack vertically data frame df1 and data frame df3
df.stacked <- rbind(df1,df3)

# but to stack vertically you need to make sure that the columns of the data frames match
try(df.stacked.bis <- rbind(df1,df2)) # you get an error, why? (different column names)
# or use rbind.fill
library(plyr)
df.stacked.bis <- rbind.fill(df1,df2)

# clearing our workspace
rm(list = ls())
closeAllConnections()

# go back to the QMF folder (it contains the data/ folder)
setwd(getOption("qmf.root"))

# one way to deal with table is to import them as data frame with R
prices <- read.csv("data/pea_price.csv")
# create a small data frame with only the head (for slow computers)
priceshead <- head(prices)
# show the class of each columns
sapply(prices,class)
# try to convert the data to numeric
test <- as.data.frame(sapply(prices,as.numeric))
testhead <- head(test)
# put some random character into the data frame
testhead[1,2] <- "eric"
# now the second column is a character, not numeric
sapply(testhead[,2],class)
# if we force this second column to numeric, then the character "eric" is forced to NA
testhead2 <- as.data.frame(sapply(testhead,as.numeric))
sapply(testhead2[,2],class)

# how to compute the return, select some columns
col1 <- as.data.frame(testhead2[,3:9])

# do the Prices at time t
Pt <- col1[2:nrow(col1),]
# or equivalently
Pt <- col1[-1,]
# do the Prices at time t-1
Pt1 <- col1[-nrow(col1),]

# you can compute
Pt / Pt1
log(Pt / Pt1)
(Pt - Pt1) / Pt1

# question: how to put dates in index and convert to Dates
rownames(prices) <- prices$Date

rownames(prices) <- as.Date(rownames(prices),"%d/%m/%Y")


closeAllConnections()
rm(list=ls())
# go back to the QMF folder (it contains the data/ folder)
setwd(getOption("qmf.root"))

# one way to deal with table is to import them as data frame with R
prices <- read.csv("data/pea_price.csv",stringsAsFactors=FALSE)
prices2 <- read.csv("data/pea_price_2.csv",stringsAsFactors=FALSE)
# here the dates have been imported as character
class(prices$Date)
# but also the prices have been imported as character
class(prices$FR0005854700.ISIN)
# to check for the classes of all columns
sapply(prices,class)
# we might want to merge both data frame (while skipping the first column of the second data frames)
prices <- data.frame(prices,prices2[-1])
# we can release the second data frame from the memory
rm(prices2)
# we can flip the data frame vertically to have the rows in increase dates
prices <- prices[nrow(prices):1, ]
# before converting everything to numeric type, we keep the information on dates
datesdf <- prices$Date
# convert all data to numeric
prices <- as.data.frame(sapply(prices, as.numeric))
# put the dates back
prices$Date <- datesdf
rm(datesdf)
# we convert the Date to date format
prices$Date <- as.Date(prices$Date,"%d/%m/%Y")

# keep columns with no NA
prices <- prices[colSums(is.na(prices))==0]

# plot a random selection of 7 items
library(ggplot2)
library(reshape)
simplechantillon <- sample(prices[,2:ncol(prices)], 7, replace = FALSE)  # 7 distinct columns
# add the dates
simplechantillon['Date'] <- prices$Date
mdf<-reshape::melt(simplechantillon,id.vars="Date")  # explicit: data.table also defines melt()
ggplot(data=mdf,aes(x=Date,y=value)) + geom_line(aes(color=variable),linewidth=1.25)+scale_x_date("Year")+scale_y_continuous("Share prices")


# empty memory
rm(list = ls())

# ============================================================================
# Work with data.table ----
# ============================================================================
library(data.table)
library(stringr)

# create a simple data table
dt <- data.table(data.frame(Source=c("A","A","A","B","B","C"),Target=c("C","D","E","C","D","E"),Weight=c(3,7,8,10,11,90)))
sapply(dt,class)
setkey(dt,Target)
# imagine we wish to multiply the common Target weights of A and B, only the common
dt[(dt$Source=="A")|(dt$Source=="B"),prod(Weight)*(length(Weight)-1),by=Target]

# A lengthier way to achieve the same outcome
# all possible combinations between source and target
allcombin <- unique(CJ(dt$Source,dt$Target))
colnames(allcombin) <- c("Source","Target")
allcombin <- allcombin[,Weight := 0]
setkey(allcombin,Target)
dt <- merge(dt,allcombin,by=c("Source","Target"),all.y=TRUE)
dt <- dt[,Weight := Weight.x +  Weight.y]
dt[is.na(Weight), Weight := 0,by=Weight]
outcomelengthy <- dt[(dt$Source=="A")|(dt$Source=="B"),prod(Weight),by=Target]
sum(outcomelengthy$V1)

# we do the same as above but with data tables
prices <- fread("data/pea_price.csv")
prices2 <- fread("data/pea_price_2.csv")
# all items have been imported as characters
class(prices$Date)
class(prices$`FR0005854700 ISIN`)
# you need to indicate the key of your data frame, typically for time series the key is the date
setkey(prices,Date)
setkey(prices2,Date)
# merge both data
prices <- prices[prices2,]
rm(prices2)
# we can flip the data frame vertically to have the rows in increase dates
prices <- prices[nrow(prices):1, ]
# before converting everything to numeric type, we keep the information on dates
datesdf <- prices$Date

# convert all data to numeric
prices[, names(prices) := lapply(.SD, as.numeric)]
class(prices$`FR0005854700 ISIN`)
# get back the dates information
prices$Date <- datesdf
rm(datesdf)
# if we want to compute the mean price per year as an exercise
prices[,year:=str_sub(Date,7,11)]
prices[,mean(`FR0005854700 ISIN`,na.rm = TRUE),by=year]

# For more information on data.table:
# https://cran.r-project.org/web/packages/data.table/vignettes/datatable-intro.html

# empty memory
rm(list = ls())

# ============================================================================
# Lists ----
# ============================================================================

# List are similar to vectors except that they are recursively formed; 
# you can construct list of lists
# The list() operator allow to keep the types of the different items you add to the list
v <- c(1,c(1,1),"A")
l <- list(1,c(1,1),"A")
v
l
# Return c(1,1) as a list
l[2]
# Return c(1,1) as a itself (a vector)
l[[2]]

# Illustration of a use of lists

A <- matrix(c("RNO","SG","BNP"))
B <- matrix(c(1.1,1.2,1.3))

res <- list(A,B)
res[[1]]
res[[2]]

# clearing our workspace
rm(list = ls())

# ============================================================================
# Exercise 2: loading data from external files ----
# ============================================================================
# Exercise : 
# 1: create a function that give you log (x- threshold)
# 2: use tryCatch to return a NA in case there is a warning
# 3: apply your function to the house prices in Farmer 2015
# 4: for the NA in your create list, replace with the value of the previous result

data <- read.csv(file = "data/farmer_2015.csv", sep = ";")

myfunc <- function(x){return(log(x - thr))}

thr <- 5
myfunc(6)
x <- 7
y <- 6
log(-1)
try(log("oops"))  # error: non-numeric argument

tryCatch(log(x - thr),
         warning = function(w) {;log(y - thr)},
         error = function(e) {;log(y - thr)})

test <- sapply(data[,2], myfunc)

library(zoo)
test <- zoo(test)
test <- na.locf(test)

plot(test)

head(data)      


# what do we know about 'data'
# gives us the dimensions as nrow, ncol
dim(data)

# gives the column names of data
names(data)

# we can use the $ to access a named column of our object
# by typing objectName$columnName
data$house_price
# gives us the length of a vector
length(data$house_price)

# loading data from an R library
# use the lalonde data set in the Matching library
data(lalonde)
ls()
names(lalonde)
dim(lalonde)
lalonde$age




# ============================================================================
# Basic commands: sequences, names, matrices ----
# ============================================================================

# Making Objects:
# sequence

# a sequence from 0 to 5 in 0.5 increments
seq.1 <- seq(0, 5, 0.5)
seq.1
class(seq.1)

# a sequence of integers
seq2 <- 1:10
seq2
class(seq2)

# making squares
myX <- 1:4
myX

# note: for an array, ^2 will to element wise operation
myX <- myX^2
myX

# assign some names to our array using "c()", which concatenates
names(myX)
names(myX) <- c("1 squared", "2 squared", "3 squared", "4 squared")
myX
class(myX) 


# matrices
mat1 <- matrix(data = 1:10, nrow = 5, ncol = 2, byrow = TRUE)
mat1
mat2 <- matrix(1:10, 5, 2, byrow = FALSE)
mat2


# element wise logical comparison
# == will test for equality in the two compared objects
mat1 == mat2

ls()
x == y
x == x

# clearing our workspace
rm(list = ls())
ls()

# ============================================================================
# Data frames, again: indexing and subsetting ----
# ============================================================================
# a data frame is a list of variables of the same length with unique row names.
# it is best to put your data in a data frame when working with it
# when you read in data, R coerces it into a data.frame class

data <- read.csv(file = "data/farmer_2015.csv", sep= ";")
class(data)
names(data)
rownames(data)

# we might want to transform the column Date into dates and set as index
data$Date <- as.Date(zoo::as.yearqtr(data$Date, format = "%YQ%q"))
rownames(data) <- data$Date
data$Date <- NULL

# we can also make our own data frame
x <- 1:16
y <- seq(1, 4, .2)
length(x)
length(y)
myData <- data.frame( X = x, Y = y )
names(myData)
dim(myData)
class(myData)

myData$X
myData$Y

# attach allows us to access the columns of a data frame without typing the object name and run some diagnostics
attach(myData)
X
mean(X)
min(X)
Y
max(Y)
sd(Y)
detach(myData)


# ============================================================================
# Indexing (super important) ----
# ============================================================================

# indexing arrays:
x
# return the first five elements of x
x[1:5]
# set the first five elements of x to NA
x[1:5] <- NA
x
# check for missing values, returns a logical vector
is.na(x)
class(is.na(x))

# putting a ! infront of a logical command will negate it
!is.na(x)

# find the number of missing values and non-missing values
sum(is.na(x))
sum(!is.na(x))

# check that all elements are accounted for
sum(is.na(x)) + sum(!is.na(x)) == length(x)

# making a new vector from a subset of an existing vector
# in this case, make a vector of all non-missing values of x
x2 <- x[!is.na(x)]
x2
length(x2)
# in this case, drop the first three values
x3 <- x[-c(1:3)]
x3

# indexing matrices:
# same as before, except now we must include a row and column value
data

# are there any missing data points in data$x1
sum(is.na(data$house_price))
# what about in the whole data set
sum(is.na(data))

# element row = 2, column = 1
data[2, 1]

# rows in which houseprice < 4.6
data[data$house_price < 4.6, ]

# rows in which x < 4.6 and only columns 1 to 3
data[data$house_price < 4.6, 1:3]

attach(data)
data[house_price < 4.6, 1:3]
detach(data)

# indexing variables
# find all values of y1 for which u_transformed is also negative
data$ln_stock[data$u_transformed < 0]


# ============================================================================
# Finding subsets with logical conditions ----
# ============================================================================
# note: we have >, <, >=, <=, !=, == as logical operators to use
# we can use & to do and statements and | to do or statements
library(MASS)
data(painters)
names(painters)
rownames(painters)

attach(painters)

painters$School
painters[painters$School == "A", ]
rownames(painters[painters$School == "A", ])

# the & means that the row must meet both conditions
# putting nothing after the comma means "all columns"
painters[School == "A" & Drawing < 15, ]
subPainters <- painters[School == "A" & Drawing < 15, ]
subPainters

colours <- painters[School == "A" & Drawing >= 15, 3]
colours

painters[School == "A" | School == "B", ]
painters[School == "C" | School == "D", c(1:3)]

# putting nothing before the comma means "all rows"
painters[, 2]

detach(painters)

# ============================================================================
# String manipulation: paste() and stringr ----
# ============================================================================
# 'paste()' is helpful
#  great for text data, naming things, searching over variables 

vstring=painters[1:3,1]   
paste(vstring[1:3],'-',sep='') 
paste(vstring[1],'-',vstring[2],'-',vstring[3],'-',sep='')

# 'stringr' package is very useful for manipulating string objects in R       
library(stringr)
str_length(paste(vstring[1],'-',vstring[2],'-',vstring[3],'-',sep=''))
str_sub(paste(vstring[1],'-',vstring[2],'-',vstring[3],'-',sep=''),1,5)  


# ============================================================================
# For loops, again ----
# ============================================================================
# we can use for loops to iterate through a process
# we index with i, and the for loop iterates through the process from the starting value to
# the last value, in this case starting at 1, incrementing by 1 each time, until we reach 5
for( i in 1:5) {
  print(paste("Surfs up!  I'm on iteration", i))
}

for( i in 5:1) {
  print(paste("Surfs down!  I'm on iteration", i))
}

# in this case we don't plug in consecutive values for i
for( i in c(3,10,9)) {
  print(paste("Bummer!  I've got i equal to", i))
}

# making squares using a for loop
squares <- NULL
for( i in 1:5) {
  squares <- c(squares, i^2)	
  print(squares)
  print(paste("The mean is currently", mean(squares)))
}
squares
mean(squares)

# dealing with columns of data
data
for( i in 1:ncol(data)) {
  print(paste("The mean of column", i, "is", mean(data[, i]) ))
}

# find the max for each of the rows of data using a for loop
# note: the object we index by doesn't have to be 'i', that is just convention
for( j in 1:nrow(data)) {
  print(paste("The max of row", j, "is", max(data[j, 2:ncol(data)]) ))
}




# ============================================================================
# apply() over rows and columns ----
# ============================================================================
# apply is similar to the idea of a for loop, but it is tailored to matrices and data frame
help(apply)
# we must pass in our object, the margin we want to iterate over (1 means rows, 2 means columns), and a function

# find the means of all the columns of data
means <- apply(data[,2:ncol(data)], 2, mean)
means

# find the max of all the rows of data
rowMax <- apply(data, 1, max)
rowMax



# ============================================================================
# Linear regression with lm() ----
# ============================================================================
data(cars)
attach(cars)

help(lm)

# make a new variable, speed^2
speed2 = speed^2

# model 1, just speed on distance with intercept
lm1 = lm(dist ~ speed)
summary(lm1)

# model 2, speed and speed^2 on distance with intercept
lm2 = lm(dist ~ speed + speed2)
summary(lm2)

# model 3, speed and speed^2 no intercept
lm3 = lm(dist ~ speed + speed2 - 1)
summary(lm3)

detach(cars)       

# ============================================================================
# Sequences and matrix algebra: t(), %*%, solve() ----
# ============================================================================
# additional operators
# : sequence operator
# counts whole numbers from n to m; used in looping for instance
1:9                                                             

# integer sequences
seq(from=1,to=9,by=.7)

# matrix operations     
dim(data)[2] == dim(data)[1]
databis <- data[,2:ncol(data)]

X = as.matrix(databis[,1:(dim(databis)[2]/2)])
Y = as.matrix(databis[,(dim(databis)[2]/2+1):dim(databis)[2]])
dim(X)
dim(Y)

# Transpose
t(X)      
# Matrix multiplication
t(Y)%*%X      
# inverse
solve(t(Y)%*%X)

# ============================================================================
# Using a package: temporal disaggregation with tempdisagg ----
# ============================================================================
# (Chow-Lin method, see the lecture notes)
library(tempdisagg)

help(td)

data(swisspharma)
# one indicator, no intercept
mod1 <- td(sales.a ~ 0 + exports.q)
summary(mod1)
plot(mod1)

# disaggregated and true series
ts.plot(cbind(predict(mod1), sales.q), col = c("red", "blue"),
        main = "Quarterly Sales, Pharmaceuticals and Chemicals")
legend("topleft", c("Estimated values", "True values"), 
       lty = 1, col = c("red", "blue"), inset = 0.05)

?read.csv
