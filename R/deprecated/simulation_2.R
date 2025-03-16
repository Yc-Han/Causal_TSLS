###############################################################################
# 0) PACKAGES
###############################################################################
library(tidyverse)
library(caret)
library(pROC)
library(randomForest)
library(xgboost)
library(lmtest)
set.seed(42)

###############################################################################
# 1) SIMULATE DATA
###############################################################################
n <- 3000

# 1.1 Generate X
X1_cont <- rnorm(n)
X2_cont <- rnorm(n)
X3_cont <- rnorm(n)
X4_cont <- rnorm(n)

X1 <- cut(X1_cont, breaks = quantile(X1_cont, seq(0,1,0.25)), 
          include.lowest=TRUE, labels=c("Q1","Q2","Q3","Q4"))
X2 <- cut(X2_cont, breaks = quantile(X2_cont, seq(0,1,0.25)), 
          include.lowest=TRUE, labels=c("Q1","Q2","Q3","Q4"))
X3 <- cut(X3_cont, breaks = quantile(X3_cont, seq(0,1,0.25)), 
          include.lowest=TRUE, labels=c("Q1","Q2","Q3","Q4"))
X4 <- cut(X4_cont, breaks = quantile(X4_cont, seq(0,1,0.25)), 
          include.lowest=TRUE, labels=c("Q1","Q2","Q3","Q4"))

# 1.2 "True" probability for Z=1
X1_num <- as.numeric(X1)
X2_num <- as.numeric(X2)
X3_num <- as.numeric(X3)
X4_num <- as.numeric(X4)
X_conf <- rnorm(n, 0, 3)

logitZ <- -1 + 0.5 * (X1_num * X2_num) +
  0.5 * X3_num - 0.3 * X4_num + 0.3 * (X3_num * X4_num) -
  0.1 * (X1_num * X2_num * X3_num * X4_num) + X_conf - 0.1*X_conf^2 + 0.1*X_conf^3

pZ <- plogis(logitZ)
Z  <- rbinom(n, size=1, prob=pZ)
Z_factor <- factor(Z, levels=c(0,1), labels=c("Class1","Class2"))

mean(Z)

lmZ <- lm(Z ~ factor(X1) + factor(X2) + factor(X3) + factor(X4))
summary(lmZ)
resettest(lmZ)
resettest(lmZ, type = "regressor")

# 1.4 Potential D(0), D(1)
## sample another confounder used in D() and Y()
X_conf2 <- rnorm(n, 1, 2)
pD0 <- plogis(
  -2.0 - 0.3*(X1_num) + 0.4*(X1_num*X2_num) * X_conf2
  + 0.3*(X3_num*X4_num) * X_conf2
  - 0.1*(X1_num*X3_num*X4_num) + X_conf2 - 0.1*X_conf2^2 + 0.01*X_conf2^3
)
pD1 <- plogis(
  -1  # bigger intercept 
  + 0.6*(X1_num) + 0.2*(X2_num) + 0.2*(X1_num*X2_num) * X_conf2
  + 0.1*(X3_num*X4_num) * X_conf2
  + 0.1*(X1_num*X2_num*X4_num) - X_conf2 + 0.1*X_conf2^2 - 0.01*X_conf2^3
)
mean(pD1 - pD0)
pD1 <- pmax(pD1, pD0)

U  <- runif(n)
D0 <- as.numeric(U < pD0)
D1 <- as.numeric(U < pD1)
D_numeric <- ifelse(Z==1, D1, D0)
D_factor  <- factor(D_numeric, levels=c(0,1), labels=c("Class1","Class2"))

###############################################################################
# 2) PART 1: CLASSIFY Z ~ (X1,X2,X3,X4), MULTI-MODEL COMPARISON
###############################################################################
df_z <- data.frame(
  Z = Z_factor,  # factor with levels c("Class1","Class2")
  X1 = as.factor(X1),
  X2 = as.factor(X2),
  X3 = as.factor(X3),
  X4 = as.factor(X4)
)
df_z$rowIndex <- seq_len(nrow(df_z))

ctrl_z <- trainControl(
  method          = "cv",
  number          = 10,
  savePredictions = "all",
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,  # so caret can compute e.g. AUC
  verboseIter     = FALSE
)
# We'll treat "Class2" as the "positive" class for AUC
# In caret, we can set "classProbs = TRUE" and "summaryFunction=twoClassSummary",
#   but must specify the "Class2" as the positive in factor levels:
levels(df_z$Z)
# Ensure "Class2" is recognized as the "positive" if needed:
# (By default, twoClassSummary uses the last factor level as "positive", so we are good.)

set.seed(42)

# 2.1 Model 1: Logistic
glm_model_z <- train(
  Z ~ X1 * X2 * X3 * X4,
  data      = df_z,
  method    = "glm",
  family    = "binomial",
  metric    = "ROC",
  trControl = ctrl_z
)

# 2.2 Model 2: Random Forest
rf_model_z <- train(
  Z ~ X1 + X2 + X3 + X4,
  data      = df_z,
  method    = "rf",
  metric    = "ROC",
  trControl = ctrl_z,
  ntree     = 1000,
  tuneGrid  = data.frame(mtry = c(2,3,4))
)

# 2.3 Model 3: XGB
xgb_grid <- expand.grid(
  nrounds          = c(500, 1000, 1500),
  max_depth        = c(3, 6),
  eta              = c(0.1, 0.01),
  gamma            = c(0),
  colsample_bytree = c(1),
  min_child_weight = c(1),
  subsample        = c(1)
)
xgb_model_z <- train(
  Z ~ X1 + X2 + X3 + X4,
  data      = df_z,
  method    = "xgbTree",
  metric    = "ROC",
  trControl = ctrl_z,
  tuneGrid  = xgb_grid
)

# 2.4 Compare models
results_z <- resamples(list(
  glm = glm_model_z,
  rf  = rf_model_z,
  xgb = xgb_model_z
))
summary(results_z)    # shows AUC, Sens, Spec etc. across folds
bwplot(results_z, metric = "ROC")
dotplot(results_z, metric = "ROC")
# Extract OOF preds
# We'll call them 'rf_oof'
oof_df <- rf_model_z$pred
# If there's tuning, subset to bestTune:
best <- rf_model_z$bestTune
for(nm in names(best)) {
  oof_df <- oof_df[oof_df[[nm]] == best[[nm]], ]
}
# Average over multiple lines if needed
library(dplyr)
rf_oof <- oof_df %>%
  group_by(rowIndex) %>%
  summarize(rf_oof = mean(Class2))
df_z$lm_in_sample <- predict(lmZ, newdata = df_z)
df_z$rowIndex <- seq_len(nrow(df_z))
df_z <- merge(df_z, rf_oof, by="rowIndex", all.x=TRUE)

# Define EztildeX = (rf_oof - lm_in_sample)
df_z$EztildeX <- df_z$rf_oof - df_z$lm_in_sample

###############################################################################
# 3) PART 2: ESTIMATE P(AT|X), P(NT|X), P(CP|X) = P(D=1|Z=0) etc.
#    using OOF predictions but *without* unobserved type.
###############################################################################
# In real data, we only see (D,Z,X). We do not see type = C, AT, NT, Df.
# We want to fit:   D ~ X + Z   => p(D=1|X,Z).
# Then define:
#   p(AT|X) = p(D=1|X,Z=Class1)
#   p(NT|X) = 1 - p(D=1|X,Z=Class2)
#   p(CP|X) = p(D=1|X,Z=Class2) - p(D=1|X,Z=Class1)
###############################################################################
df_d <- data.frame(
  D  = D_factor,
  Z  = Z_factor,
  X1 = X1,
  X2 = X2,
  X3 = X3,
  X4 = X4,
  pD0 = pD0,
  pD1 = pD1
)
df_d$rowIndex <- seq_len(n)

###############################################################################
# 3) MULTI-MODEL: OBTAIN OOF PREDICTIONS FOR D ~ Z + X
#    AND EXTRACT p(D=1|Z=Class1), p(D=1|Z=Class2)
###############################################################################

# We will do a custom K-fold loop, but inside each training fold
# we use caret to fit each model type (Logistic, RF, XGB) similarly
# to how you did in part 1, then we predict.

K <- 10
set.seed(999)
fold_idx <- createFolds(df_d$D, k=K, list=TRUE)

# To store out-of-fold predictions for each model:
oof_pred <- data.frame(
  rowIndex = df_d$rowIndex,
  pD1_zC1_glm = NA_real_,  # logistic predictions for Z=Class1
  pD1_zC2_glm = NA_real_,
  pD1_zC1_rf  = NA_real_,  # randomForest predictions
  pD1_zC2_rf  = NA_real_,
  pD1_zC1_xgb = NA_real_,  # xgb predictions
  pD1_zC2_xgb = NA_real_
)

# Define a simple tune grid for each model
# (You can expand these as you wish)
rf_grid <- data.frame(mtry = c(2,3))
xgb_grid <- expand.grid(
  nrounds          = c(300, 600),
  max_depth        = c(3, 6),
  eta              = c(0.1),
  gamma            = c(0),
  colsample_bytree = c(1),
  min_child_weight = c(1),
  subsample        = c(1)
)

ctrl_d <- trainControl(
  method          = "cv",    # or "repeatedcv"
  number          = 3,       # an *inner* CV for the model's tuning if you like
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,
  verboseIter     = FALSE,
  savePredictions = "final", # just final model among the tuning grid
  allowParallel   = FALSE    # turn on if you have parallel backend
)

for(k in seq_len(K)) {
  test_ids  <- fold_idx[[k]]
  train_ids <- setdiff(seq_len(nrow(df_d)), test_ids)
  
  # Subset train / test folds
  d_train <- df_d[train_ids, ]
  d_test  <- df_d[test_ids, ]
  
  #############################################################################
  # 3.1) Fit GLM (Logistic) in caret
  #############################################################################
  glm_fit <- train(
    D ~ Z + X1 + X2 + X3 + X4,
    data      = d_train,
    method    = "glm",
    family    = "binomial",
    metric    = "ROC",
    trControl = ctrl_d
  )
  
  # 3.2) Fit Random Forest in caret
  rf_fit <- train(
    D ~ Z + X1 + X2 + X3 + X4,
    data      = d_train,
    method    = "rf",
    metric    = "ROC",
    ntree     = 500,     # or 1000
    tuneGrid  = rf_grid,
    trControl = ctrl_d
  )
  
  # 3.3) Fit XGB in caret
  xgb_fit <- train(
    D ~ Z + X1 + X2 + X3 + X4,
    data      = d_train,
    method    = "xgbTree",
    metric    = "ROC",
    tuneGrid  = xgb_grid,
    trControl = ctrl_d
  )
  
  # For each row in d_test, create 2 'counterfactual' copies:
  d_test_C1 <- d_test %>% mutate(Z = factor("Class1", levels=c("Class1","Class2")))
  d_test_C2 <- d_test %>% mutate(Z = factor("Class2", levels=c("Class1","Class2")))
  
  # Predict p(D=1) for both 'Z=Class1' and 'Z=Class2'
  # (In caret, the predicted probability of the second factor level is named after that level.
  #  Since D has levels = c("Class1","Class2"), the probability of "Class2" is p(D=1).)
  glm_pC1 <- predict(glm_fit, newdata=d_test_C1, type="prob")[,"Class2"]
  glm_pC2 <- predict(glm_fit, newdata=d_test_C2, type="prob")[,"Class2"]
  
  rf_pC1  <- predict(rf_fit,  newdata=d_test_C1, type="prob")[,"Class2"]
  rf_pC2  <- predict(rf_fit,  newdata=d_test_C2, type="prob")[,"Class2"]
  
  xgb_pC1 <- predict(xgb_fit, newdata=d_test_C1, type="prob")[,"Class2"]
  xgb_pC2 <- predict(xgb_fit, newdata=d_test_C2, type="prob")[,"Class2"]
  
  # Store them in oof_pred:
  oof_pred$pD1_zC1_glm[test_ids] <- glm_pC1
  oof_pred$pD1_zC2_glm[test_ids] <- glm_pC2
  oof_pred$pD1_zC1_rf[test_ids]  <- rf_pC1
  oof_pred$pD1_zC2_rf[test_ids]  <- rf_pC2
  oof_pred$pD1_zC1_xgb[test_ids] <- xgb_pC1
  oof_pred$pD1_zC2_xgb[test_ids] <- xgb_pC2
}

###############################################################################
# 4) COMPUTE p(AT|X), p(NT|X), p(CP|X) FOR EACH MODEL, THEN MSE vs. TRUTH
###############################################################################
# Merge OOF predictions with df_d
df_d_pred <- df_d %>%
  left_join(oof_pred, by="rowIndex")

# Define a small helper:
mse <- function(a, b) mean((a - b)^2)

# For each model M in {glm,rf,xgb}, define:
#   pAT_hat(M) = pD1_zC1_M
#   pNT_hat(M) = 1 - pD1_zC2_M
#   pCP_hat(M) = pD1_zC2_M - pD1_zC1_M
# Compare to pAT_true = pD0, pNT_true=1-pD1, pCP_true=(pD1-pD0)

# ------------------ GLM ------------------
df_d_pred <- df_d_pred %>%
  mutate(
    pAT_glm = pD1_zC1_glm,
    pNT_glm = 1 - pD1_zC2_glm,
    pCP_glm = pD1_zC2_glm - pD1_zC1_glm
  )

cat("GLM MSE(AT):", mse(df_d_pred$pAT_glm, df_d_pred$pD0), "\n")
cat("GLM MSE(NT):", mse(df_d_pred$pNT_glm, 1 - df_d_pred$pD1), "\n")
cat("GLM MSE(CP):", mse(df_d_pred$pCP_glm, df_d_pred$pD1 - df_d_pred$pD0), "\n")

# ------------------ RF -------------------
df_d_pred <- df_d_pred %>%
  mutate(
    pAT_rf = pD1_zC1_rf,
    pNT_rf = 1 - pD1_zC2_rf,
    pCP_rf = pD1_zC2_rf - pD1_zC1_rf
  )

cat("RF  MSE(AT):", mse(df_d_pred$pAT_rf, df_d_pred$pD0), "\n")
cat("RF  MSE(NT):", mse(df_d_pred$pNT_rf, 1 - df_d_pred$pD1), "\n")
cat("RF  MSE(CP):", mse(df_d_pred$pCP_rf, df_d_pred$pD1 - df_d_pred$pD0), "\n")

# ------------------ XGB ------------------
df_d_pred <- df_d_pred %>%
  mutate(
    pAT_xgb = pD1_zC1_xgb,
    pNT_xgb = 1 - pD1_zC2_xgb,
    pCP_xgb = pD1_zC2_xgb - pD1_zC1_xgb
  )

cat("XGB MSE(AT):", mse(df_d_pred$pAT_xgb, df_d_pred$pD0), "\n")
cat("XGB MSE(NT):", mse(df_d_pred$pNT_xgb, 1 - df_d_pred$pD1), "\n")
cat("XGB MSE(CP):", mse(df_d_pred$pCP_xgb, df_d_pred$pD1 - df_d_pred$pD0), "\n")

###############################################################################
# 4) PART C: Compute Weights Using OOF EztildeX & Compare to "True" Weights
###############################################################################
# Suppose you have "type" in {AT,C,NT,Df} and the "true" weights:
#   w_at, w_cp, w_nt   (already computed from your formulas)
# We'll store them in a data.frame to merge with the OOF EztildeX.
df_weights <- data.frame(
  rowIndex = seq_len(n),
  type     = type,
  w_AT_true = NA,
  w_C_true  = NA,
  w_NT_true = NA
)
df_weights$w_AT_true[df_weights$type=="AT"] <- df_d_pred$pAT_xgb[df_weights$type=="AT"]
df_weights$w_C_true [df_weights$type=="C"]  <- df_d_pred$pCP_xgb[df_weights$type=="C"]
df_weights$w_NT_true[df_weights$type=="NT"] <- df_d_pred$pNT_xgb[df_weights$type=="NT"]

# Merge in EztildeX from df_z
df_weights <- df_weights %>%
  left_join(select(df_z, rowIndex, EztildeX), by="rowIndex")

# Also we need pZ, pD0, pD1, and Z from the environment or a merged data frame
df_weights <- df_weights %>%
  left_join(select(df_d, rowIndex, pD0, pD1), by="rowIndex")

# For "mean(tildeZ * D)", replicate your approach:
tildeZ_D_mean <- mean( (Z - lmZ$fitted.values)*D_numeric)  # or however you define

# Then define the "estimated" weights w_AT_hat etc. using EztildeX
#   w_AT_hat = EztildeX * pD0 / tildeZ_D_mean for type=AT only
df_weights$w_AT_hat <- df_weights$EztildeX * df_weights$pD0 / tildeZ_D_mean
df_weights$w_AT_hat[df_weights$type != "AT"] <- NA

# Similarly for Compliers:
#   w_C_hat = (EztildeX + LM*(1-pZ)) * (pD1 - pD0)/tildeZ_D_mean
#   But pZ must come from your environment or be merged; e.g. df_weights$pZ
#   Below we skip pZ if you haven't stored it, or adapt as needed
# Example if you do have it:
# df_weights$w_C_hat <- (df_weights$EztildeX + df_weights$lm_in_sample*(1-df_weights$pZ)) * 
#   (df_weights$pD1 - df_weights$pD0)/ tildeZ_D_mean
# df_weights$w_C_hat[df_weights$type != "C"] <- NA

# For NT:
df_weights$w_NT_hat <- df_weights$EztildeX * (1 - df_weights$pD1) / tildeZ_D_mean
df_weights$w_NT_hat[df_weights$type!="NT"] <- NA

# Finally compare to w_AT_true etc.
mse_w <- function(a,b) mean((a - b)^2, na.rm=TRUE)
cat("MSE w(AT):", mse_w(df_weights$w_AT_hat, df_weights$w_AT_true), "\n")
# and so forth for w_C_hat, w_NT_hat if you've defined them

summary(df_weights$w_AT_hat - df_weights$w_AT_true)
