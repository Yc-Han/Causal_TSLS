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

# Example: define a discrete grouping factor from some X variables
df$Xgroup <- with(df, interaction(expbin, black, south, smsa, drop=TRUE))
# Show how many levels
table(df$Xgroup)

df_sum <- df %>%
  group_by(Xgroup) %>%
  summarise(
    # denominator for each g: sum of fractional membership within this Xgroup
    denom_cp = sum(cp),
    denom_at = sum(at),
    denom_nt = sum(nt),
    
    # E[Z_tilde * D | g, Xgroup]
    E_ZtT_cp = sum(cp * Z_tilde * D) / sum(cp),
    E_ZtT_at = sum(at * Z_tilde * D) / sum(at),
    E_ZtT_nt = sum(nt * Z_tilde * D) / sum(nt),
    
    # E[Z_tilde | g, Xgroup]
    E_Zt_cp  = sum(cp * Z_tilde) / sum(cp),
    E_Zt_at  = sum(at * Z_tilde) / sum(at),
    E_Zt_nt  = sum(nt * Z_tilde) / sum(nt),
    
    # Probability of (g, x) in the sample, i.e. E[I(Xgroup=x)* p_g(i)]
    p_cp     = sum(cp) / nrow(df),
    p_at     = sum(at) / nrow(df),
    p_nt     = sum(nt) / nrow(df)
  )
# across df_sum, fill NA with 0
df_sum[is.na(df_sum)] <- 0

df_sum <- df_sum %>%
  mutate(
    # For g = C, example:
    # E[I(T=1) Z_tilde | C, x] = E_ZtT_cp 
    # p(C,x) = p_cp
    # multiply them, then scale by 1/E[Z_tilde*T] in a separate step
    w1_cp = E_ZtT_cp * p_cp,
    w1_at = E_ZtT_at * p_at,
    w1_nt = E_ZtT_nt * p_nt
  )

EZT <- mean(df$Z_tilde * df$D)  # This is \hat{E}[Z_tilde*T]

df_sum <- df_sum %>%
  mutate(
    w1_cp = w1_cp / EZT,
    w1_at = w1_at / EZT,
    w1_nt = w1_nt / EZT
  )


df_cate <- df %>%
  group_by(Xgroup) %>%
  summarise(
    # Sum of complier weights in this cell
    denom_cp_1 = sum(cp * D),
    denom_cp_0 = sum(cp * (1 - D)),
    
    # Weighted average of Y for "treated" portion
    E_Y1_cp = ifelse(denom_cp_1 > 0,
                     sum(cp * D * Y) / denom_cp_1, NA),
    
    # Weighted average of Y for "untreated" portion
    E_Y0_cp = ifelse(denom_cp_0 > 0,
                     sum(cp * (1 - D) * Y) / denom_cp_0, NA),
    
    # CATE_cp = difference
    CATE_cp = E_Y1_cp - E_Y0_cp,
    
    # ... similarly for always takers ...
    denom_at_1 = sum(at * D),
    denom_at_0 = sum(at * (1 - D)),
    E_Y1_at = ifelse(denom_at_1 > 0,
                     sum(at * D * Y) / denom_at_1, NA),
    E_Y0_at = ifelse(denom_at_0 > 0,
                     sum(at * (1 - D) * Y) / denom_at_0, NA),
    CATE_at = E_Y1_at - E_Y0_at,
    
    # ... and never takers ...
    denom_nt_1 = sum(nt * D),
    denom_nt_0 = sum(nt * (1 - D)),
    E_Y1_nt = ifelse(denom_nt_1 > 0,
                     sum(nt * D * Y) / denom_nt_1, NA),
    E_Y0_nt = ifelse(denom_nt_0 > 0,
                     sum(nt * (1 - D) * Y) / denom_nt_0, NA),
    CATE_nt = E_Y1_nt - E_Y0_nt
  )
df_cate[is.na(df_cate)] <- 0

df_join <- left_join(df_sum, df_cate, by = "Xgroup") %>%
  select(Xgroup, CATE_cp, CATE_at, CATE_nt, w1_cp, w1_at, w1_nt) %>%
  mutate(
    beta_cp = CATE_cp * w1_cp,
    beta_at = CATE_at * w1_at,
    beta_nt = CATE_nt * w1_nt,
    beta_rich = beta_cp + beta_at + beta_nt
  )
sum(df_join$beta_rich)
