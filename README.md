# Homework 07: Machine learning (Julia Du)

Detailed instructions for this homework assignment are [here](https://cfss.uchicago.edu/homework/machine-learning/).

## Executing the files

For Part 1 (analyzing student debt), you can find the source code at [scorecard.Rmd](scorecard.Rmd) and the rendered report at [scorecard.md](scorecard.md). 
Using scorecard data, this looks at various models to predict student debt, including basic linear regression, linear regression with cross-validation, and a tuned decision tree. 

For Part 2 (analyzing attitudes towards racist professors), you can find the source code at [gss-colrac.Rmd](gss-colrac.Rmd) and the rendered report at [gss-colrac.md](gss-colrac.md).
Using GSS data, we try to predict attitudes towards racist college professors with logistic regression, random forest, 5-nearest-neighbors, ridge logistic, and tuned boosted tree models. 

There's nothing special about running these files - just open the .Rmd files & knit them to get the .md rendered reports.

## Required packages

You should have the following packages installed:

```r
library(reprex)
library(tidyverse)
library(knitr)
library(lubridate)
library(tidymodels)
library(rcfss)

library(usemodels)
library(kknn)
library(glmnet)
library(xgboost)
library(vip)

```
2nd chunk of packages are more specialized - most of these are needed for running the specific models we use.
