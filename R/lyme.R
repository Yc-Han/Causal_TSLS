library(AER)
library(haven)
library(tidyverse)
library(lmtest)
library(sandwich)
library(ddml)
library(ranger)

# Load the data from lyme/
lyme <- read.csv("data/lyme2018.csv")
lyme <- lyme[complete.cases(lyme),]

### Define Variables: without Year
Y <- lyme$LogLyme0103_1113
Z <- lyme$DensityResIndex
D <- lyme$PercWuiPop
X <- lyme %>%
  select(Year, PercLandForest, EdgeDensity, MeanPatchArea) %>%
  as.matrix()
X1 <- X[, 1]
X2 <- X[, 2]
X3 <- X[, 3]
X4 <- X[, 4]

# OLS
lr <- lm(Y ~ D + X, data = lyme)
summary(lr)
cluster_se1 <- vcovCL(lr, cluster = ~AgStatDistrict)
coeftest(lr, cluster_se1)

# 2SLS, no covariates
iv1 <- ivreg(Y ~ D | Z, data = lyme)
summary(iv1, diagnostics = TRUE)
cluster_se2 <- vcovCL(iv1, cluster = ~AgStatDistrict)
coeftest(iv1, cluster_se2)

# 2SLS, with covariates
iv2 <- ivreg(Y ~ X + D | X + Z, data = lyme)
summary(iv2, diagnostics = TRUE)
cluster_se3 <- vcovCL(iv2, cluster = ~AgStatDistrict)
coeftest(iv2, cluster_se3)

regZ1 <- lm(Z ~ X, data = lyme)
summary(regZ1)
resettest(regZ1)
resettest(regZ1, type = "regressor")

# check first stage
fs <- lm(D ~ X1 + X2 + X3 + X4 + Z, data = lyme)
summary(fs)
fs0 <- lm(D ~ Z, data = lyme)
summary(fs0)
# likelihood ratio test
lrtest(fs0, fs)

# check multicollinearity
vif(fs)
Dhat <- fs$fitted.values
# check second stage
ss <- lm(Y ~ Dhat + X1 + X2 + X3 + X4, data = lyme)
summary(ss)
ss0 <- lm(Y ~ Dhat, data = lyme)
summary(ss0)
lrtest(ss0, ss)
vif(ss)

# 2SLS, with flexible terms, i.e. d=3 polynomials
iv3 <- ivreg(Y ~ D + X1 + poly((poly(X2, 2) * poly(X3, 2) * poly(X4, 2)), 3) | X1 + poly((poly(X2, 2) * poly(X3, 2) * poly(X4, 2)), 3) + Z, data = lyme)
summary(iv3, diagnostics = TRUE)
cluster_se4 <- vcovCL(iv3, cluster = ~AgStatDistrict)
coeftest(iv3, cluster_se4)
lmZ_flex <- lm(Z ~ X1 + poly((poly(X2, 2) * poly(X3, 2) * poly(X4, 2)), 3), data = lyme)
summary(lmZ_flex)
resettest(lmZ_flex)
resettest(lmZ_flex, type = "regressor")
library(splines)
lmZ_splines <- lm(Z ~ X1 * bs(X2, 3) * bs(X3, 3) * bs(X4, 3), data = lyme)
summary(lmZ_splines)
resettest(lmZ_splines)
resettest(lmZ_splines, type = "regressor")
iv4 <- ivreg(Y ~ D + X1 + (bs(X2, 3) + bs(X3, 3) + bs(X4, 3)) | X1 + (bs(X2, 3) + bs(X3, 3) + bs(X4, 3)) + Z, data = lyme)
summary(iv4, diagnostics = TRUE)
# DDML
set.seed(123)
learners_multiple <- list(
                          list(fun = mdl_xgboost,
                               args = list(nrounds = 1000,
                                           max_depth = 6)),
                          list(fun = mdl_ranger,
                               args = list(num.trees = 1000,
                                           mtry = 3)))
pliv1 <- ddml_pliv(Y, D, Z, X,
                            learners = learners_multiple,
                            ensemble_type = c('nnls','nnls1', 'ols','singlebest', 'average'),
                            shortstack = TRUE,
                            sample_folds = 100,
                            cv_folds = 100,
                            silent = FALSE)
summary(pliv1)

