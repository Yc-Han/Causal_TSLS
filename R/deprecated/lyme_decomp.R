##### DEPRECATED! #####
##### DEPRECATED! #####
##### DEPRECATED! #####
##### DEPRECATED! #####
library(AER)
library(haven)
library(tidyverse)
library(lmtest)
library(sandwich)
library(ddml)
library(ranger)
library(xgboost)
library(caret)
library(pROC)
library(glmnet)
library(randomForest)
library(ggplot2)

# Load the data
lyme <- read.csv("data/lyme2018.csv")
lyme <- lyme[complete.cases(lyme),]
lyme$AgStatDistrict <- as.factor(lyme$AgStatDistrict)
### Define Variables
Y <- lyme$LogLyme0103_1113
Z <- lyme$DensityResIndex
lyme$Z <- Z
D <- lyme$PercWuiPop
X <- lyme %>%
  select(Year, PercLandForest, EdgeDensity, MeanPatchArea) %>%
  as.matrix()
# --- OLS ---
lr <- lm(Y ~ D + X, data = lyme)
summary(lr)
cluster_se1 <- vcovCL(lr, cluster = ~AgStatDistrict)
coeftest(lr, cluster_se1)

# --- 2SLS, no covariates ---
iv1 <- ivreg(Y ~ D | Z, data = lyme)
summary(iv1, diagnostics = TRUE)
cluster_se2 <- vcovCL(iv1, cluster = ~AgStatDistrict)
coeftest(iv1, cluster_se2)

# --- 2SLS, with covariates ---
iv2 <- ivreg(Y ~ D + X | Z + X, data = lyme)
summary(iv2, diagnostics = TRUE)
cluster_se3 <- vcovCL(iv2, cluster = ~AgStatDistrict)
coeftest(iv2, cluster_se3)

# --- Regress Z on X with a simple LM (not OOF) ---
regZ1 <- lm(Z ~ X, data = lyme)
summary(regZ1)
plot(lyme$DensityResIndex, predict(regZ1),
     xlab = "Actual Z", ylab = "Predicted Z (in-sample LM)")
resettest(regZ1)
resettest(regZ1, type = "regressor")

ztilde_in_sample <- Z - predict(regZ1)
cat("In-sample (simple LM) RMSE:", 
    sqrt(mean((Z - predict(regZ1))^2)), "\n")

# tag each row for OOF extraction
lyme$rowIndex <- seq_len(nrow(lyme))


dataset <- data.frame(X, Z = Z)

# --- Cross-Validation Setup ---
ctrl <- trainControl(
  method           = "cv",
  number           = 10,
  savePredictions  = "all",
  returnResamp     = "all",
  verboseIter      = FALSE
)

# --- Model Grids / Tuning ---
# XGBoost
xgb_grid <- expand.grid(
  nrounds          = c(100, 500, 1000),
  max_depth        = c(2, 4, 6),
  eta              = c(0.01, 0.05, 0.1),
  gamma            = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample        = 1
)
# Random Forest
rf_grid <- expand.grid(mtry = c(2, 3, 4))
# Elastic Net
enet_grid <- expand.grid(
  alpha  = c(0, 0.5, 1),
  lambda = seq(0.001, 1, length = 10)
)

# --- Train Each Model with 10-fold CV (OOF) ---
set.seed(42)
xgb_model <- train(
  Z ~ .,
  data      = dataset,
  method    = "xgbTree",
  trControl = ctrl,
  metric    = "RMSE",
  tuneGrid  = xgb_grid
)
rf_model_1000 <- train(
  Z ~ .,
  data         = dataset,
  method       = "rf",
  trControl    = ctrl,
  metric       = "RMSE",
  tuneGrid     = rf_grid,
  ntree        = 1000,
  importance   = TRUE
)
rf_model_500 <- train(
  Z ~ .,
  data         = dataset,
  method       = "rf",
  trControl    = ctrl,
  metric       = "RMSE",
  tuneGrid     = rf_grid,
  ntree        = 500,
  importance   = TRUE
)
rf_model_2000 <- train(
  Z ~ .,
  data         = dataset,
  method       = "rf",
  trControl    = ctrl,
  metric       = "RMSE",
  tuneGrid     = rf_grid,
  ntree        = 2000,
  importance   = TRUE
)
# OOF Linear Model
lm_model <- train(
  Z ~ .,
  data      = dataset,
  method    = "lm",
  trControl = ctrl,
  metric    = "RMSE"
)
# GLM
quasi_model <- train(
  Z ~ .,
  data      = dataset,
  method    = "glm",
  trControl = ctrl,
  family    = "quasi",
  metric    = "RMSE"
)

# Elastic Net
elastic_model <- train(
  Z ~ .,
  data      = dataset,
  method    = "glmnet",
  trControl = ctrl,
  metric    = "RMSE",
  tuneGrid  = enet_grid,
  family    = "gaussian"
)

# --- Compare models' cross-validated performance ---
results <- resamples(list(
  xgb      = xgb_model,
  rf_1000  = rf_model_1000,
  rf_500   = rf_model_500,
  rf_2000  = rf_model_2000,
  lm       = lm_model,
  glmquasi = quasi_model,
  enet     = elastic_model
))
summary(results)
bwplot(results, metric = "RMSE")
dotplot(results, metric = "RMSE")

# --- Compute OOF RMSE from the resampled predictions ---
compute_oof_rmse <- function(model) {
  df_oof <- model$pred %>%
    group_by(rowIndex) %>%
    summarise(
      obs  = mean(obs),
      pred = mean(pred)
    )
  sqrt(mean((df_oof$obs - df_oof$pred)^2))
}

xgb_oof_rmse    <- compute_oof_rmse(xgb_model)
rf_1000_oof_rmse <- compute_oof_rmse(rf_model_1000)
rf_500_oof_rmse <- compute_oof_rmse(rf_model_500)
rf_2000_oof_rmse <- compute_oof_rmse(rf_model_2000)
lm_oof_rmse     <- compute_oof_rmse(lm_model)
quasi_oof_rmse  <- compute_oof_rmse(quasi_model)
enet_oof_rmse   <- compute_oof_rmse(elastic_model)

cat("XGBoost OOF RMSE:       ", xgb_oof_rmse,   "\n")
cat("Random Forest 1000 OOF RMSE: ", rf_oof_rmse,    "\n")
cat("Random Forest 500 OOF RMSE: ", rf_500_oof_rmse,    "\n")
cat("Random Forest 2000 OOF RMSE: ", rf_2000_oof_rmse,    "\n")
cat("Linear Model OOF RMSE:  ", lm_oof_rmse,    "\n")
cat("Quasi Model OOF RMSE:   ", quasi_oof_rmse, "\n")
cat("Elastic Net OOF RMSE:   ", enet_oof_rmse,  "\n")

# --- Collect OOF predictions for each model ---
get_oof_preds <- function(model, name) {
  model$pred %>%
    group_by(rowIndex) %>%
    summarise(pred = mean(pred)) %>%
    rename(!!name := pred)
}

xgb_oof <- get_oof_preds(xgb_model,  "xgb_oof")
rf_oof  <- get_oof_preds(rf_model_500   ,   "rf_oof")
lm_oof  <- get_oof_preds(lm_model,   "lm_oof")
lyme_oof <- lyme %>%
  left_join(xgb_oof, by = "rowIndex") %>%
  left_join(rf_oof,  by = "rowIndex") %>%
  left_join(lm_oof,  by = "rowIndex")

cat("XGBoost vs. actual, OOF RMSE check:\n")
sqrt(mean((lyme_oof$Z - lyme_oof$xgb_oof)^2))
cat("RF vs. actual, OOF RMSE check:\n")
sqrt(mean((lyme_oof$Z - lyme_oof$rf_oof)^2))
cat("LM vs. actual, OOF RMSE check:\n")
sqrt(mean((lyme_oof$Z - lyme_oof$lm_oof)^2))

# --- Construct E[tildeZ | X]_OOF = (ML OOF) - (LM OOF) ---
lyme_oof <- lyme_oof %>%
  mutate(lm_z = predict(lm_model, newdata = lyme_oof)) %>%
  mutate(EztildeX = rf_oof - lm_z)

# Example comparison: XGBoost vs. LM predictions
ggplot(lyme_oof, aes(x = rf_oof, y = lm_oof)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color = "red") +
  labs(x = "Random Forest OOF (Z-hat)", y = "Linear Model OOF (Z-hat)") +
  theme_bw()

# Example: Mean and SD of E[ztilde|X] by AgStatDistrict (XGBoost version)
means_eztildeX <- lyme_oof %>%
  group_by(AgStatDistrict) %>%
  summarise(
    mean = mean(EztildeX, na.rm = TRUE),
    sd   = sd(EztildeX,   na.rm = TRUE)
  ) %>%
  arrange(mean)

# Re-factor for plotting in ascending mean
means_eztildeX <- means_eztildeX %>%
  mutate(AgStatDistrict = factor(AgStatDistrict, levels = AgStatDistrict[order(mean)]))

ggplot(means_eztildeX, aes(x = mean, y = AgStatDistrict)) +
  geom_point() +
  geom_errorbar(aes(xmin = mean - sd, xmax = mean + sd), width = 0.2) +
  labs(x = "Mean EztildeX (XGBoost - LM)", 
       y = "AgStatDistrict", 
       title = "Mean and SD of EztildeX by AgStatDistrict") +
  geom_vline(xintercept = 0, linetype = "dashed", color = "red") +
  theme_bw()