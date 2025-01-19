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
summary(ivreg(Y ~ X + D | X + Z))

regZ <- lm(Z ~ X)
resettest(regZ)

### Estimation IV with saturation
X1 <- X[, 1]
X2 <- X[, 2]
X3 <- X[, 3]
X4 <- X[, 4]
Z_int <- cbind(Z, Z * X1, Z * X2, Z * X3, Z * X4, Z * X1 * X2,
               Z * X1 * X3, Z * X1 * X4, Z * X2 * X3, Z * X2 * X4,
               Z * X3 * X4, Z * X1 * X2 * X3, Z * X1 * X2 * X4,
               Z * X1 * X3 * X4, Z * X2 * X3 * X4, Z * X1 * X2 * X3 * X4)
Z_int <- cbind(Z, Z*X)

summary(ivreg(Y ~ X1*X2*X3*X4 + D | X1*X2*X3*X4 + Z))

X_full <- cbind(X, X1 * X2, X1 * X3, X1 * X4, X2 * X3, X2 * X4, X3 * X4,
                X1 * X2 * X3, X1 * X2 * X4, X1 * X3 * X4, X2 * X3 * X4,
                X1 * X2 * X3 * X4)

regZ_full <- lm(Z_int ~ X_full)
resettest(regZ_full)

### ivreg without X
summary(ivreg(Y ~ D | Z))

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
learners_multiple <- list(list(fun = ols),
                          list(fun = mdl_glmnet),
                          list(fun = mdl_ranger),
                          list(fun = mdl_xgboost))
pliv_rich <- ddml_pliv(Y, D, Z, X,
                            learners = learners_multiple,
                            ensemble_type = c('nnls', 'nnls1', 'ols','average'),
                            shortstack = TRUE,
                            sample_folds = 10,
                            silent = FALSE)
summary(pliv_rich)
# -------------- estimating LATE using DDML
late_fit_short <- ddml_late(Y, D, Z, X,
                            learners = learners_multiple,
                            ensemble_type = c('nnls', 'nnls1', 'ols','average'),
                            shortstack = TRUE,
                            sample_folds = 10,
                            silent = FALSE)
summary(late_fit_short)

# -------------- Decomposing the estimator according to Blandhol et al. 2022
# under saturated X.
# --------------
# Define choice groups
# --------------
# Estimate D on Z and X_full with random forest
library(randomForest)
set.seed(123)
D <- as.factor(D)
colnames(X_full)[colnames(X_full) == ""] <- paste0("V", seq_len(sum(colnames(X_full) == "")))
X_with_Z <- cbind(Z, X_full)
# Rename unnamed columns to something meaningful
colnames(X_with_Z)[colnames(X_with_Z) == ""] <- paste0("V", seq_len(sum(colnames(X_with_Z) == "")))
treat_rf <- randomForest(x = X_with_Z, y = D, ntree = 200)
# Extract the predicted probabilities
Xz1 <- cbind(Z = 1, X_full)
Xz0 <- cbind(Z = 0, X_full)
df$D_z1 <- predict(treat_rf, newdata = Xz1, type = "prob")[, 2]
df$D_z0 <- predict(treat_rf, newdata = Xz0, type = "prob")[, 2]
# Choice groups:
# CP: D_z1 > 0.5 & D_z0 < 0.5
# AT: D_z1 > 0.5 & D_z0 > 0.5
# NT: D_z1 < 0.5 & D_z0 < 0.5
df$choice_group <- ifelse(df$D_z1 > 0.5& df$D_z0 < 0.5, "CP",
                          ifelse(df$D_z1 > 0.5 & df$D_z0 > 0.5, "AT", "NT"))
# plot joint distribution of D_z1 and D_z0
ggplot(df, aes(x = D_z0, y = D_z1, color = choice_group)) +
  geom_point() + theme_bw() + xlab("P(D = 1 | Z = 0)") + ylab("P(D = 1 | Z = 1") +
  ggtitle("Choice groups")

# too strict, use fractional, it is also equal to P(g,x)
df$cp <- df$D_z1 - df$D_z0
df$at <- df$D_z1
df$nt <- 1 - df$D_z0

# --------------
# Z_tilde
# --------------
# Regression on Z
regZ <- lm(Z ~ X_full)
df$Z_tilde <- residuals(regZ)

### Cov(D, Z | x)

df_cov <- df %>%
  group_by(Xgroup) %>%
  summarise(
    Ztilde = Z_tilde,
    cov_DZ = cov(D, Z),
    pr_cp_x = sum(cp) / nrow(df),
    pr_at_x = sum(at) /nrow(df),
    pr_nt_x = sum(nt) /nrow(df),
    term_cp = cov_DZ * pr_cp_x / Ztilde,
    term_at = cov_DZ * pr_at_x / Ztilde,
    term_nt = cov_DZ * pr_nt_x / Ztilde
  )
sum(df_cov$term_cp)
sum(df_cov$term_at)
sum(df_cov$term_nt)
