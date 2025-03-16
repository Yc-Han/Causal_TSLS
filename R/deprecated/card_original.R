library(AER)
library(haven)
library(tidyverse)
library(grf)
library(ddml)
library(ranger)

### Data
card <- read_dta("https://raw.github.com/scunning1975/mixtape/master/card.dta")
suppressMessages(attach(card))

Y <- lwage
D <- educ
expsq <- expersq / 100
X <- cbind(exper, expsq, black, south, smsa, reg661, reg662, reg663, reg664,
           reg665, reg666, reg667, reg668, smsa66)
Z <- nearc4
df <- data.frame(Y, D, X, Z) %>%
  na.omit()

# -------------- OLS regerssion of Y on D and X
ols_reg <- lm(Y ~ D + X)
# summary(ols_reg)
# coeff of D
coefficients(ols_reg)[2]

# -------------- estimating TSLS (selection on unobservables & observables)
# first stage
fs <- lm(D ~ X + Z)
# store first-stage slope coefficient
fsCoef <- coef(fs)["Z"]
# reduced form
rf <- lm(Y ~ X + Z)
# store reduced-form slope coefficient
rfCoef <- coef(rf)["Z"]
# divide reduced-form coefficient by first-stage coefficient
rfCoef / fsCoef
# alternative way of estimating LATE
summary(ivreg(Y ~ D + X | Z + X), diagnostics = TRUE)

# ivreg without X
summary(ivreg(Y ~ D | Z), diagnostics = TRUE)

# -------------- estimating beta_rich using DDML

set.seed(123)
learners_multiple <- list(list(fun = ols),
                          list(fun = mdl_glmnet),
                          list(fun = mdl_ranger),
                          list(fun = mdl_xgboost))
pliv_rich <- ddml_pliv(Y, D, Z, X,
                            learners = learners_multiple,
                            ensemble_type = c('nnls', "nnls1", 'ols',
                                              'average'),
                            shortstack = TRUE,
                            sample_folds = 10,
                            silent = FALSE)
summary(pliv_rich)

# -------------- estimating weights
covariates <- setdiff(names(df), c("Y", "D", "Z"))
set.seed(123)
formula_D <- as.formula(
  paste("D ~ ", paste(c("Z", covariates), collapse = " + "))
)
model_D <- ranger(formula_D, data = df)
z_values <- c(0, 1)
pred_array <- array(
  NA_real_,
  dim = c(nrow(df), model_D$num.trees, length(z_values))
)
for (i in seq_along(z_values)) {
  df_z <- transform(df, Z = z_values[i])
  preds <- predict(model_D, data = df_z, predict.all = TRUE)$predictions
  pred_array[,, i] <- preds
}
is_non_decreasing <- function(vec) {
  all(diff(vec) >= 0)
}
mono_matrix <- matrix(0L, nrow = nrow(df), ncol = model_D$num.trees)
for (r in seq_len(nrow(df))) {
  for (b in seq_len(model_D$num.trees)) {
    pred_vec <- pred_array[r, b, ]
    if (is_non_decreasing(pred_vec)) {
      mono_matrix[r, b] <- 1
    }
  }
}
mono_prob <- rowMeans(mono_matrix)
df$mono <- mono_prob
boxplot(mono_prob, main = "Monotonicity Probability")
sum(mono_prob < 0.5) / nrow(df)

set.seed(123)
formula_Z <- as.formula(
  paste("Z ~", paste(covariates, collapse = " + "))
)
model_Z <- ranger(formula_Z, data = df, probability = TRUE)

pZ_hat <- predict(model_Z, data = df, type = "response")$predictions[,2]
vZ_hat <- pZ_hat * (1 - pZ_hat)
df$VarZ <- vZ_hat
# 6) Final weights = mono_prob * var(Z|X), normalized to sum to 1
w_raw <- mono_prob * vZ_hat
w_final <- w_raw / sum(w_raw)
df$w_final <- w_final

# explore dependence of w_final on covariates
# 1. ranger with feature importance
set.seed(123)
formula_w <- as.formula(
  paste("w_final ~", paste(covariates, collapse = " + "))
)
model_w <- ranger(formula_w , data = df, importance = "permutation")
importance(model_w)
# 2. kmeans clustering
set.seed(123)
K <- 16
km_out <- kmeans(X, centers = K, nstart = 20)
df$cluster <- km_out$cluster

cluster_summary <- df %>%
  group_by(cluster) %>%
  summarize(
    avg_varZ = mean(VarZ),
    avg_mono = mean(mono),
    sum_w = sum(w_final),
    avg_w = mean(w_final),
    n = n(),
    # average of each covariate
    across(covariates, ~ round(mean(.x), 3))
  )

# -------------- Decomposing the estimator according to Blandhol et al. 2022

# Regression on Z
regZ <- lm(Z ~ X)
summary(regZ)
# histogram of fitted values
# ggplot(data.frame(fitted(regZ)), aes(x = fitted(regZ))) +
#  geom_histogram(bins = 20) + xlab("Fitted values of Z ~ X") + theme_bw()
# RESET test for rich covariates
resettest(regZ)

# the scalar E[tilde(Z)^2]^-1
scalar_ztilde <- 1 / mean((Z - fitted(regZ))^2)
cat("E[tilde(Z)^2]^-1: ", scalar_ztilde)

# -------------------------------------------------------
# Estimate Xi_k_x
# -------------------------------------------------------
fs_int <- lm(D ~ X * Z, data = df)
summary(fs_int)
df$p0 <- predict(fs_int,
                 newdata = data.frame(X = X, Z = 0))
df$p1 <- predict(fs_int,
                 newdata = data.frame(X = X, Z = 1))
df <- df %>%
  # define xi_0(x_i) and xi_1(x_i) by ordering p(0,x_i), p(1,x_i)
  mutate(
    xi_0 = if_else(p0 <= p1, 0, 1),
    xi_1 = if_else(p0 <= p1, 1, 0)
  )

summary(df$xi_0)
summary(df$xi_1)

# -------------------------------------------------------
# Obtain t_tilde_1(x) for each observation under Z = 0 and Z = 1
# -------------------------------------------------------

df$hat_t1 <- predict(fs,
                     newdata = data.frame(X = X, Z = df$xi_1))
df$hat_t0 <- predict(fs,
                     newdata = data.frame(X = X, Z = df$xi_0))
df$tilde_t1_x <- df$hat_t1 - df$hat_t0


summary(df$tilde_t1_x)

# -------------------------------------------------------
# Obtain Upsilon_1(x) for each observation
# -------------------------------------------------------
# For E[Y | X, Z],
# we perform a regression of Y on X and Z
# and use fitted values as E[Y | X, Z]
mod_Y <- lm(Y ~ X * Z, data = df)

df$E_Y_X_Z1 <- predict(mod_Y,
                       newdata = data.frame(X = X, Z = df$xi_1))
df$E_Y_X_Z0 <- predict(mod_Y,
                       newdata = data.frame(X = X, Z = df$xi_0))
df$Upsilon1_x <- df$E_Y_X_Z1 - df$E_Y_X_Z0

summary(df$Upsilon1_x)

# -------------------------------------------------------
# Obtain phi_1(x) for each observation
# -------------------------------------------------------
# For P(Z=1|X), we do logistic
mod_logit <- glm(Z ~ X, family = binomial(link="logit"))
df$pZ1_X  <- predict(mod_logit, type="response") # logistic p-hat
df$phi1_x <- df$pZ1_X * (1 - df$pZ1_X) * scalar_ztilde
summary(df$phi1_x)

# -------------------------------------------------------
# Obtain beta_rich
# -------------------------------------------------------
# beta_rich = mean[Upsilon_1(x) * tilde_t1(x) * pZ1_X * (1 - pZ1_X) * scalar_ztilde]
df <- df %>%
  mutate(
    local_term = Upsilon1_x * tilde_t1_x * phi1_x,
    local_term_plus = Upsilon1_x * tilde_t1_x * (tilde_t1_x > 0) * phi1_x,
    local_term_minus = Upsilon1_x * tilde_t1_x * (tilde_t1_x < 0) * phi1_x
  )
mean(df$local_term)
mean(df$local_term_plus)
mean(df$local_term_minus)

# -------------------------------------------------------
# Weighting
# -------------------------------------------------------
beta_rich_pos <- mean(df$local_term_plus)

### rho_1(x) = E[T|Z=1,X=x] - E[T|Z=0,X=x]

### beta_rich_pos_cor := beta_rich_pos / mean[rho_1(x) * t_tilde_1(x) * phi_1(x)]

beta_rich_pos_cor <- beta_rich_pos / mean((df$p1 - df$p0) * df$tilde_t1_x * (df$tilde_t1_x > 0) * df$phi1_x)
beta_rich_pos_cor

# -------------------------------------------------------
# Generalized Random Forest as in Athey et al. 2019
# -------------------------------------------------------
set.seed(123)
# Split the data into training and testing sets
data_clean <- na.omit(data.frame(Y, D, Z, X))
Y <- as.vector(data_clean$Y)
D <- as.vector(data_clean$D)
Z <- as.vector(data_clean$Z)
X <- as.matrix(data_clean[, -(1:3)])  # Covariates matrix


train_indices <- sample(1:nrow(X), size = 0.9 * nrow(X))  # 90% for training
X_train <- X[train_indices, ]
X_test <- X[-train_indices, ]
Y_train <- Y[train_indices]
Y_test <- Y[-train_indices]
D_train <- D[train_indices]
D_test <- D[-train_indices]
Z_train <- Z[train_indices]
Z_test <- Z[-train_indices]

# Train the Instrumental Forest model
set.seed(123)  # For reproducibility

inst_forest <- instrumental_forest(X = X_train, Y = Y_train, W = D_train,
                                   Z = Z_train, tune.parameters = "all", num.trees = 2000)
                                   #num.trees = 1000,
                                   #min.node.size = 10, imbalance.penalty = 0.2)
scores <- get_scores(inst_forest)
# plot the scores
summary(scores)
hist(scores, breaks = 30, col = "lightblue", main = "Scores Distribution")

# Predict treatment effects for the testing set
predictions <- predict(inst_forest, X_test, estimate.variance = TRUE)

# Extract predicted treatment effects and confidence intervals
tau_hat <- predictions$predictions
tau_var <- predictions$variance.estimates
tau_ci_lower <- tau_hat - 1.96 * sqrt(tau_var)
tau_ci_upper <- tau_hat + 1.96 * sqrt(tau_var)

# Set a threshold for acceptable confidence interval width (e.g., CI width <= 2)
ci_width <- tau_ci_upper - tau_ci_lower
threshold <- 2  # Adjust this based on the distribution of CI widths

# Filter out unreliable points
reliable_points <- ci_width <= threshold
sum(reliable_points)

# Subset the reliable predictions
reliable_tau_hat <- tau_hat[reliable_points]
reliable_ci_lower <- tau_ci_lower[reliable_points]
reliable_ci_upper <- tau_ci_upper[reliable_points]

# Plot only the reliable points
plot(reliable_tau_hat, ylim = c(-0.05, 0.5), pch = 16, col = "blue",
     xlab = "Test Sample Index", ylab = "CATE")
arrows(1:length(reliable_tau_hat), reliable_ci_lower, 1:length(reliable_tau_hat),
       reliable_ci_upper, length = 0.05, angle = 90, code = 3, col = "red")
abline(h = 0, lty = 2)
# add mean ATE as text
text(1, 0.45, paste("ATE:", round(average_treatment_effect(inst_forest)[1], 4)),
     pos = 4, col = "red")
text(1, 0.425, paste("ATE (SD):", round(average_treatment_effect(inst_forest)[2], 2)),
     pos = 4, col = "red")

title("Predicted Treatment Effects with 95% CI")

causal_grf <- causal_forest(X = X_train, Y = Y_train, W = D_train,
                            tune.parameters = "all",
                            num.trees = 2000)
average_treatment_effect(causal_grf)
test_calibration(causal_grf)

tau.hat <- predict(causal_grf)$predictions
high.effect <- tau.hat > median(tau.hat)
ate.high <- average_treatment_effect(causal_grf, subset = high.effect)
ate.low <- average_treatment_effect(causal_grf, subset = !high.effect)
ate.high[["estimate"]] - ate.low[["estimate"]] +
  c(-1, 1) * qnorm(0.975) * sqrt(ate.high[["std.err"]]^2 + ate.low[["std.err"]]^2)
