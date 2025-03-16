library(AER)
library(haven)
library(tidyverse)
library(ddml)

### Data
card <- read_dta("https://raw.github.com/scunning1975/mixtape/master/card.dta")
suppressMessages(attach(card))

Y <- lwage
D <- as.numeric(educ > 13)
expbin <- as.numeric(exper > 8)
X <- cbind(expbin, black, south, smsa)
Z <- nearc4
idx_complete <- complete.cases(Y, D, X, Z)
Y <- Y[idx_complete]
D <- D[idx_complete]
X <- X[idx_complete, ]
Z <- Z[idx_complete]

df <- data.frame(Y, D, Z, X)

### Estimation OLS
ols_reg <- lm(Y ~ D + X)
summary(ols_reg)
coefficients(ols_reg)[2]

### Estimation IV, with covariates
# first stage
fs <- lm(D ~ X + Z)
summary(fs)
# test relevance condition
coeftest(fs)
# store first-stage slope coefficient
fsCoef <- coef(fs)["Z"]
# reduced form
rf <- lm(Y ~ X + Z)
summary(rf)
# test exclusion restriction: Z not correlated with error term
err <- residuals(rf)
cor.test(Z, err)

# store reduced-form slope coefficient
rfCoef <- coef(rf)["Z"]
# divide reduced-form coefficient by first-stage coefficient
rfCoef / fsCoef
# alternative way of estimating LATE
summary(ivreg(Y ~ X + D | X + Z), diagnostics = TRUE)

regZ <- lm(Z ~ X)
resettest(regZ)
resettest(regZ, type = "regressor")
### Estimation IV with saturation
X1 <- X[, 1]
X2 <- X[, 2]
X3 <- X[, 3]
X4 <- X[, 4]
X_full <- cbind(X, X1 * X2, X1 * X3, X1 * X4, X2 * X3, X2 * X4, X3 * X4,
                X1 * X2 * X3, X1 * X2 * X4, X1 * X3 * X4, X2 * X3 * X4,
                X1 * X2 * X3 * X4)

Z_int <- cbind(Z, Z * X_full)
# saturated
summary(ivreg(Y ~ X1*X2*X3*X4 + D | X1*X2*X3*X4 + Z), diagnostics = TRUE)
# sw
summary(ivreg(Y ~ X1*X2*X3*X4 + D | X1*X2*X3*X4 + Z_int), diagnostics = TRUE)

regZ_full <- lm(Z_int ~ X_full)
resettest(regZ_full)
resettest(regZ_full, type = "regressor")
### ivreg without X
summary(ivreg(Y ~ D | Z), diagnostics = TRUE)

### check homogeneous treatment effect
summary(lm(Y ~ D * X1 * X2 * X3 * X4))
# Fit the full model:
mod_full <- lm(Y ~ D * X1 * X2 * X3 * X4)
# Fit the reduced model (no D interactions, i.e., only main effect of D):
mod_reduced <- lm(Y ~ D + X1 + X2 + X3 + X4)
# Compare models:
anova(mod_reduced, mod_full)

### Estimation DDML
# -------------- estimating beta_rich using DDML
set.seed(123)
# Estimate the local average treatment effect using short-stacking with base
#     learners ols, rlasso, and xgboost.
learners_multiple <- list(
  list(fun = mdl_xgboost,
       args = list(nrounds = 1000,
                   max_depth = 6)),
  list(fun = mdl_ranger,
       args = list(num.trees = 1000,
                   mtry = 3)))
pliv_rich <- ddml_pliv(Y, D, Z, X,
                            learners = learners_multiple,
                            ensemble_type = c('nnls','nnls1', 'ols','singlebest', 'average'),
                            shortstack = TRUE,
                            sample_folds = 100,
                            silent = FALSE)
summary(pliv_rich)
# -------------- estimating LATE using DDML
late_fit_short <- ddml_late(Y, D, Z, X,
                            learners = learners_multiple,
                            ensemble_type = c('nnls','nnls1', 'ols','singlebest', 'average'),
                            shortstack = TRUE,
                            sample_folds = 100,
                            silent = FALSE)
summary(late_fit_short)
