library(AER)
library(haven)
library(tidyverse)
library(lmtest)
library(sandwich)
library(ddml)

# Load the data from lyme/
lyme <- read.csv("lyme/lyme2018.csv")
lyme <- lyme[complete.cases(lyme),]

### Define Variables: without Year
Y <- lyme$LogLyme0103_1113
Z <- lyme$DensityResIndex
D <- lyme$PercWuiPop
X <- lyme %>%
  select(PercLandForest, EdgeDensity, MeanPatchArea) %>%
  as.matrix()

# OLS
lr <- lm(Y ~ D + X, data = lyme)
summary(lr)
cluster_se1 <- vcovCL(lr, cluster = ~AgStatDistrict)
coeftest(lr, cluster_se1)

# 2SLS, no covariates
iv1 <- ivreg(Y ~ D | Z, data = lyme)
summary(iv1)
cluster_se2 <- vcovCL(iv1, cluster = ~AgStatDistrict)
coeftest(iv1, cluster_se2)

# 2SLS, with covariates
iv2 <- ivreg(Y ~ X + D | X + Z, data = lyme)
summary(iv2)
cluster_se3 <- vcovCL(iv2, cluster = ~AgStatDistrict)
coeftest(iv2, cluster_se3)

regZ1 <- lm(Z ~ X, data = lyme)
resettest(regZ1)

# DDML
set.seed(123)
learners_multiple <- list(list(fun = ols),
                          list(fun = mdl_glmnet),
                          list(fun = mdl_ranger),
                          list(fun = mdl_xgboost))
pliv1 <- ddml_pliv(Y, D, Z, X,
                            learners = learners_multiple,
                            ensemble_type = c('nnls', 'nnls1', 'ols','average'),
                            shortstack = TRUE,
                            sample_folds = 5,
                            cv_folds = 5,
                            silent = FALSE)
summary(pliv1)

### in X, include Year, and do the same thing again
X <- lyme %>%
  select(Year, PercLandForest, EdgeDensity, MeanPatchArea) %>%
  as.matrix()

lr2 <- lm(Y ~ D + X, data = lyme)
cluster_se4 <- vcovCL(lr2, cluster = ~AgStatDistrict)
coeftest(lr2, cluster_se4)

iv3 <- ivreg(Y ~ D | Z, data = lyme)
cluster_se5 <- vcovCL(iv3, cluster = ~AgStatDistrict)
coeftest(iv3, cluster_se5)

iv4 <- ivreg(Y ~ X + D | X + Z, data = lyme)
cluster_se6 <- vcovCL(iv4, cluster = ~AgStatDistrict)
coeftest(iv4, cluster_se6)

regZ2 <- lm(Z ~ X, data = lyme)
resettest(regZ2)

set.seed(123)
pliv2 <- ddml_pliv(Y, D, Z, X,
                            learners = learners_multiple,
                            ensemble_type = c('nnls', 'nnls1', 'ols','average'),
                            shortstack = TRUE,
                            sample_folds = 5,
                            cv_folds = 5,
                            silent = FALSE)
summary(pliv2)

### dichotomize D and Z.
# first see distribution of D and Z
plot(hist(D))
plot(hist(Z))
median(D)
median(Z)
D <- as.numeric(D > median(D))
Z <- as.numeric(Z > 0)

lr3 <- lm(Y ~ D + X, data = lyme)
cluster_se7 <- vcovCL(lr3, cluster = ~AgStatDistrict)
coeftest(lr3, cluster_se7)

iv5 <- ivreg(Y ~ D | Z, data = lyme)
cluster_se8 <- vcovCL(iv5, cluster = ~AgStatDistrict)
coeftest(iv5, cluster_se8)

iv6 <- ivreg(Y ~ X + D | X + Z, data = lyme)
cluster_se9 <- vcovCL(iv6, cluster = ~AgStatDistrict)
coeftest(iv6, cluster_se9)

regZ3 <- lm(Z ~ X, data = lyme)
resettest(regZ3) # p-value = 0.0001!

set.seed(123)
pliv3 <- ddml_pliv(Y, D, Z, X,
                            learners = learners_multiple,
                            ensemble_type = c('nnls', 'nnls1', 'ols','average'),
                            shortstack = TRUE,
                            sample_folds = 5,
                            cv_folds = 5,
                            silent = FALSE)
summary(pliv3)

set.seed(123)
late <- ddml_late(Y, D, Z, X,
                  learners = learners_multiple,
                  ensemble_type = c('nnls', 'nnls1', 'ols' , 'average'),
                  shortstack = TRUE,
                  sample_folds = 5,
                  cv_folds = 5,
                  silent = FALSE)
summary(late)
