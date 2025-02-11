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
X_full <- cbind(X, X1 * X2, X1 * X3, X1 * X4, X2 * X3, X2 * X4, X3 * X4,
                X1 * X2 * X3, X1 * X2 * X4, X1 * X3 * X4, X2 * X3 * X4,
                X1 * X2 * X3 * X4)

Z_int <- cbind(Z, Z * X_full)

summary(ivreg(Y ~ X1*X2*X3*X4 + D | X1*X2*X3*X4 + Z))

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

### estimate the possible weights on beta_pliv.

library(ranger)
estimate_rich_weights <- function(df, z_vars = "Z", plot=TRUE) {
  set.seed(123)
  # 1) Identify covariates: all columns except Y, D, Z
  covariates <- setdiff(names(df), c("Y", "D", z_vars))
  
  # 2) Ensure D, Z are factors (binary for this example)
  df$D <- as.factor(df$D)
  df$Z <- as.factor(df$Z)
  
  # 3) Fit model for D ~ Z + X
  formula_D <- as.formula(
    paste("D ~ ", paste(c(z_vars, covariates), collapse = " + "))
  )
  model_D <- ranger(formula_D, data = df, probability = TRUE)
  
  # 4) Predict P(D=1) for Z=1 vs Z=0 to get p_diff = P[T(1)>T(0)|X]
  p1_hat <- predict(model_D, data = transform(df, Z=1), type = "response")$predictions[,2]
  p0_hat <- predict(model_D, data = transform(df, Z=0), type = "response")$predictions[,2]
  p_diff <- p1_hat - p0_hat  # estimated compliance propensity
  
  # 5) Fit model for Z ~ X to get pZ_hat, then var(Z|X) = pZ_hat*(1 - pZ_hat)
  formula_Z <- as.formula(
    paste("Z ~", paste(covariates, collapse = " + "))
  )
  model_Z <- ranger(formula_Z, data = df, probability = TRUE)
  
  pZ_hat <- predict(model_Z, data = df, type = "response")$predictions[,2]
  vZ_hat <- pZ_hat * (1 - pZ_hat)
  
  # 6) Final weights = p_diff * var(Z|X), normalized to sum to 1
  w_raw <- p_diff * vZ_hat
  w_final <- w_raw / sum(w_raw)
  df$w_final <- w_final
  
  # 7) Create a 'group' label for every unique combination of X
  df$group <- apply(df[, covariates, drop = FALSE], 1, function(xx) {
    # Exclude variables matching the pattern "V<number>"
    vars_to_include <- names(xx)[xx == 1 & !grepl("^V[0-9]+.*$", names(xx))]
    paste(vars_to_include, collapse = ",")
  })
  df$group[df$group == ""] <- "none"  # Label groups with no variables set to 1
  
  # 8) Sum weights by group
  group_summary <- aggregate(w_final ~ group, data = df, sum)
  group_summary$obs_count <- aggregate(w_final ~ group, data = df, length)$w_final  # Count observations
  group_summary$sign <- ifelse(group_summary$w_final >= 0, ">=0", "<0")  # Positive/negative sign
  
  
  if (plot) {
    pt <- ggplot(group_summary, aes(x = reorder(group, w_final), y = w_final, fill = sign)) +
      geom_bar(stat = "identity") +
      geom_text(
        aes(label = obs_count),
        hjust = -0.1,  # Position labels slightly to the right of the bars
        size = 3
      ) +
      coord_flip() +  # Flip axes for better readability
      scale_fill_manual(values = c(">=0" = "blue", "<0" = "red")) +
      labs(
        title = "Sum of Weights by Subpopulation (with Observation Counts)",
        x = "Covariate Combinations",
        y = "Sum of Weights",
        fill = "Weight Sign"
      ) +
      theme_bw() +
      theme(
        axis.text.y = element_text(size = 10),
        plot.title = element_text(hjust = 0.5)
      )
  }
  print(pt)
  # Return a list with the augmented df and group‐weight table
  return(list(
    df_with_weights = df,
    group_summary = group_summary
  ))
}

# Apply the function to the card data
df_rich <- estimate_rich_weights(df)
df_full <- data.frame(Y, D, Z, X_full)
df_rich_full <- estimate_rich_weights(df_full)


### Saturate and weight
df_sw <- data.frame(Y, D, Z_int, X_full)
z_vars <- colnames(df_sw)[3:18]
df_sw_rich <- estimate_rich_weights(df_sw, z_vars = z_vars)


### in df, remove: black&smsa=1 and others=0, or expbin,black,south=1 and others=0,
### or expbin,south=1 and others=0, or black,south,smsa=1 and others=0
df_comp <- df %>%
  filter(!(black == 1 & smsa == 1 & expbin == 0 & south == 0),
         !(expbin == 1 & black == 1 & south == 1 & smsa == 0),
         !(expbin == 1 & south == 1 & black == 0 & smsa == 0),
         !(black == 1 & south == 1 & smsa == 1 & expbin == 0))
Y <- df_comp$Y
D <- df_comp$D
Z <- df_comp$Z
X_comp <- df_comp %>% select(-Y, -D, -Z) %>% as.matrix()
# redo PLIV
set.seed(123)
pliv_comp <- ddml_pliv(Y, D, Z, X_comp,
                       learners = learners_multiple,
                       ensemble_type = c('nnls', 'nnls1', 'ols','average'),
                       shortstack = TRUE,
                       sample_folds = 10,
                       silent = FALSE)
summary(pliv_comp)
### We replicated LATE!!!
