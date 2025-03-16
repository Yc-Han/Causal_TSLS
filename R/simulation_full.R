library(AER)
library(lmtest)
library(ddml)
library(tidyverse)
library(caret)
library(pROC)
library(randomForest)
library(xgboost)

set.seed(42)

### 1) SIMULATE DATA ----
n <- 3000

# 1.1 Generate X
X1_cont <- rnorm(n, mean = 0, sd = 1)
X2_cont <- rnorm(n, mean = 0, sd = 1)
X3_cont <- rnorm(n, mean = 0, sd = 1)
X4_cont <- rnorm(n, mean = 0, sd = 1)

# Binning into quartiles
X1 <- cut(X1_cont, breaks=quantile(X1_cont, seq(0,1,0.25)), 
          include.lowest=TRUE, labels=c("Q1","Q2","Q3","Q4"))
X2 <- cut(X2_cont, breaks=quantile(X2_cont, seq(0,1,0.25)), 
          include.lowest=TRUE, labels=c("Q1","Q2","Q3","Q4"))
X3 <- cut(X3_cont, breaks=quantile(X3_cont, seq(0,1,0.25)), 
          include.lowest=TRUE, labels=c("Q1","Q2","Q3","Q4"))
X4 <- cut(X4_cont, breaks=quantile(X4_cont, seq(0,1,0.25)), 
          include.lowest=TRUE, labels=c("Q1","Q2","Q3","Q4"))

# Numeric versions
X1_num <- as.numeric(X1)
X2_num <- as.numeric(X2)
X3_num <- as.numeric(X3)
X4_num <- as.numeric(X4)

# Hidden confounder
X_conf <- rnorm(n, 0, 3)

# 1.2 Generate Instrument Z
logitZ <- -1 + 0.5 * (X1_num * X2_num) +
  0.5 * X3_num - 0.3 * X4_num + 0.3 * (X3_num * X4_num) -
  0.1 * (X1_num * X2_num * X3_num * X4_num)

pZ <- plogis(logitZ)
Z   <- rbinom(n, 1, pZ)
Z_factor <- factor(Z, levels=c(0,1), labels=c("Class1","Class2"))

cat("Mean(Z) =", mean(Z), "\n")

# Simple LM for Z ~ X
lmZ <- lm(Z ~ factor(X1) + factor(X2) + factor(X3) + factor(X4))
summary(lmZ)
resettest(lmZ)
# resettest(lmZ, type="regressor") # might be problematic if X are factors

# define ztilde, EztildeX
ztilde   <- Z - predict(lmZ)
EztildeX <- pZ - predict(lmZ)

# 1.3 Potential D(0), D(1)
X_conf2 <- rnorm(n, 1, 2)
pD0 <- plogis(
  -2.0 - 0.3*(X1_num) + 0.4*(X1_num*X2_num)*X_conf2 +
    0.3*(X3_num*X4_num)*X_conf2 -
    0.1*(X1_num*X3_num*X4_num) + X_conf2 - 0.1*X_conf2^2 + 0.01*X_conf2^3
)
pD1 <- plogis(
  -1 +
    0.6*(X1_num) + 0.2*(X2_num) + 0.2*(X1_num*X2_num)*X_conf2 +
    0.1*(X3_num*X4_num)*X_conf2 +
    0.1*(X1_num*X2_num*X4_num) - X_conf2 + 0.1*X_conf2^2 - 0.01*X_conf2^3
)
# ensure pD1 >= pD0
pD1 <- pmax(pD1, pD0)

U  <- runif(n)
D0 <- as.numeric(U < pD0)
D1 <- as.numeric(U < pD1)
D  <- ifelse(Z==1, D1, D0)

D_factor <- factor(D, levels=c(0,1), labels=c("Class1","Class2"))

# 1.4 Label each subject's "type"
type <- ifelse(D0==0 & D1==0, "NT",
               ifelse(D0==1 & D1==1, "AT",
                      ifelse(D0==0 & D1==1, "C", "Df")))
table(type)

# Check exogeneity via ztilde
lmZ_type <- lm(Z ~ factor(X1)+factor(X2)+factor(X3)+factor(X4) + type)
summary(lmZ_type)  # if big significance, maybe not exogenous

# 1.5 Generate Outcomes Y(0), Y(1)
Y0 <- 1 + 0.2*(X1_num * X2_num * X3_num - X4_num - X1_num*X4_num + 
                 X2_num*X3_num - X2_num*X4_num + X3_num*X4_num) +
  0.2*(X1_num + X2_num + X3_num + X4_num) - 0.1*(X1_num + X2_num + X3_num + X4_num)^2 + 
  0.01*(X1_num + X2_num + X3_num + X4_num)^3 +
  rnorm(n, 0, 3) + 1*X_conf2 - 0.1*X_conf2^3

Y1 <- 2 + 0.3*(X1_num*X2_num*X3_num + X4_num + X1_num*X2_num + 
                 X1_num*X3_num + X2_num*X3_num + X2_num*X4_num) +
  0.2*(X1_num + X2_num + X3_num + X4_num) - 0.1*(X1_num + X2_num + X3_num + X4_num)^2 + 
  0.01*(X1_num + X2_num + X3_num + X4_num)^3 +
  rnorm(n, 0, 3) + 2*X_conf2 + 0.1*X_conf2^3

Y <- ifelse(D==1, Y1, Y0)

# True LATE among compliers
true_late <- mean(Y1[type=="C"] - Y0[type=="C"])
hetero    <- sd(Y1[type=="C"] - Y0[type=="C"])
cat("True LATE:", true_late, "(sd of complier effect:", hetero, ")\n")


### 2) IV Analysis ----
# OLS
lm_fit <- lm(Y ~ factor(D) + factor(X1)+factor(X2)+factor(X3)+factor(X4))
summary(lm_fit)
# IV w/o X
iv_model_noX <- ivreg(Y ~ factor(D) | factor(Z))
summary(iv_model_noX, diagnostics=TRUE)
# check first-stage
summary(lm(D ~ factor(Z)))
summary(lm(D ~ factor(Z) + factor(X1) + factor(X2) + factor(X3) + factor(X4)))
# IV basic
iv_model <- ivreg(Y ~ factor(D) + factor(X1)+factor(X2)+factor(X3)+factor(X4)
                  | factor(Z) + factor(X1)+factor(X2)+factor(X3)+factor(X4))
summary(iv_model, diagnostics=TRUE)
# Saturated
iv_model_saturated <- ivreg(
  Y ~ factor(D) + factor(X1)*factor(X2)*factor(X3)*factor(X4)
  | factor(Z) + factor(X1)*factor(X2)*factor(X3)*factor(X4)
)
summary(iv_model_saturated, diagnostics=TRUE)
# Saturate and Weight
iv_model_sw <- ivreg(
  Y ~ factor(D) + factor(X1)*factor(X2)*factor(X3)*factor(X4)
  | factor(Z)*factor(X1)*factor(X2)*factor(X3)*factor(X4)
)
summary(iv_model_sw, diagnostics=TRUE)
# DDML
X <- cbind(X1, X2, X3, X4)
X_design <- model.matrix(~ factor(X1)+factor(X2)+factor(X3)+factor(X4))[,-1]
learners_multiple <- list(
  list(fun = mdl_xgboost,
       args = list(nrounds = 1000,
                   max_depth = 6)),
  list(fun = mdl_ranger,
       args = list(num.trees = 1000,
                   mtry = 3)))
pliv <- ddml_pliv(Y, D, Z, X,
                  learners=learners_multiple,
                  ensemble_type=c('nnls','nnls1', 'ols','singlebest', 'average'),
                  shortstack=TRUE,
                  sample_folds=100,
                  cv_folds=100,
                  silent=FALSE)
summary(pliv)
set.seed(123)
learners_multiple <- list(
  list(fun = ols),
  list(fun = mdl_glmnet,
       args = list(alpha = 0.5)),
  list(fun = mdl_xgboost,
       args = list(nrounds = 500,
                   max_depth = 4)),
  list(fun = mdl_ranger,
       args = list(num.trees = 500,
                   mtry = 3)))
late_ddml <- ddml_late(Y, D, Z, X,
                       learners=learners_multiple,
                       ensemble_type=c('nnls','nnls1', 'ols','singlebest', 'average'),
                       shortstack=TRUE,
                       sample_folds=100,
                       cv_folds=100,
                       silent=FALSE)
summary(late_ddml)

# compare
results <- c(
  true_late,
  coef(lm_fit)[2],
  coef(iv_model)[2],
  coef(iv_model_saturated)[2],
  coef(iv_model_sw)[2],
  pliv$coef[1],
  late_ddml$late[3]
)
names(results) <- c(
  "True LATE", "OLS", "IV", "IV Saturated", 
  "IV Saturated Weighted", "DDML", "DDML LATE"
)
cat("Comparison:\n"); print(results)

### 3) Compute "True" Weights per Beta-Rich ----
# w(AT) = EtildezX * pD0 * mean(ztilde*D)^-1
tildeZ_D_mean <- mean(ztilde * D)
w_at_full <- EztildeX * pD0 / tildeZ_D_mean
w_at <- w_at_full[type=="AT"]

# w(CP) = (EtildezX + predict(lmZ)*(1-pZ)) * (pD1-pD0) * mean(ztilde*D)^-1
w_cp_full <- (EztildeX + predict(lmZ)*(1-pZ)) * (pD1-pD0) / tildeZ_D_mean
w_cp <- w_cp_full[type=="C"]

# w(NT) = EtildezX * (1-pD1) * mean(ztilde*D)^-1
w_nt_full <- EztildeX * (1-pD1) / tildeZ_D_mean
w_nt <- w_nt_full[type=="NT"]

# Visualize
weights_df <- data.frame(
  type = c(rep("AT", length(w_at)), rep("C", length(w_cp)), rep("NT", length(w_nt))),
  weight = c(w_at, w_cp, w_nt)
)
p1 <- ggplot(weights_df, aes(x=weight, fill=type)) +
  geom_density(alpha=0.5) +
  theme_bw() + xlab("Weights") + ylab("Density") +
  theme(legend.position="none",
        panel.grid.major=element_blank(),
        panel.grid.minor=element_blank())
ggsave("output/sim_weights_density.svg", p1, width=5, height=5)

# Grouped by X combos
weights_full <- rep(NA, n)
weights_full[type=="AT"] <- w_at
weights_full[type=="C"]  <- w_cp
weights_full[type=="NT"] <- w_nt

data_df <- data.frame(
  Y=Y, Z=Z, D=D, 
  X1=factor(X1), X2=factor(X2), X3=factor(X3), X4=factor(X4),
  type=type, weights=weights_full
)
data_df$group <- apply(data_df[,c("X1","X2","X3","X4")],1,function(r){
  paste0("X1:",r[1],", X2:",r[2],", X3:",r[3],", X4:",r[4])
})
data_agg <- data_df %>%
  group_by(group, type) %>%
  summarize(mean_weight = mean(weights, na.rm=TRUE), .groups="drop") %>%
  mutate(rank=rank(mean_weight)) %>%
  arrange(rank)
data_agg$group <- factor(data_agg$group, levels=unique(data_agg$group))

data_varZ <- data_df %>%
  group_by(group) %>%
  summarize(varZ = var(Z), .groups="drop")

data_agg <- data_agg %>%
  left_join(data_varZ, by="group")

cor_results <- data_agg %>%
  group_by(type) %>%
  summarize(
    cor_pearson = cor(varZ, mean_weight, use="complete.obs", method="pearson"),
    cor_spearman = cor(varZ, mean_weight, use="complete.obs", method="spearman"),
    pval_pearson = cor.test(varZ, mean_weight, method="pearson")$p.value,
    pval_spearman = cor.test(varZ, mean_weight, method="spearman")$p.value,
    .groups="drop"
  )
print(cor_results)

p2 <- ggplot(data_agg, aes(y=group, x=mean_weight, fill=type)) +
  geom_col(position="stack") +
  labs(y="Covariate Combination", x="Mean Weight", fill="Treatment Type") +
  theme_bw() +
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        axis.ticks.y=element_blank(),
        axis.text.y=element_blank(),
        axis.text.x=element_text(size=8))+
  theme(legend.position="top")
ggsave("output/sim_weights_grouped.svg", p2, width=5, height=5)

### 4) OOF for Z ~ X, define EztildeX = (rf_oof - LM) ----
cat("\n--- Out-of-Fold Classification for Z ~ X (XGBoost) ---\n")

df_z <- data.frame(
  Z = factor(ifelse(Z==1,"Class2","Class1"), levels=c("Class1","Class2")), 
  X1=factor(X1), X2=factor(X2), X3=factor(X3), X4=factor(X4)
)
df_z$rowIndex <- seq_len(n)

ctrl_z <- trainControl(
  method="cv", number=10, savePredictions="all",
  classProbs=TRUE, summaryFunction=twoClassSummary
)
set.seed(999)
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
# Extract OOF prob(Class2)
bestt <- xgb_model_z$bestTune
df_oof <- xgb_model_z$pred
for(nm in names(bestt)){
  df_oof <- df_oof[df_oof[[nm]]==bestt[[nm]],]
}
rf_oof <- df_oof %>%
  group_by(rowIndex) %>%
  summarize(rf_oof=mean(Class2), .groups="drop")

# Merge in-sample LM
df_z$lm_in_sample <- predict(lmZ, newdata=df_z)
df_z <- df_z %>% left_join(rf_oof, by="rowIndex")
df_z$EztildeX_hat <- df_z$rf_oof - df_z$lm_in_sample

# compare how much better the XGB is than the LM
cat("XGB vs. LM (in-sample):\n")
cat("Correlation:\n")
print(cor(df_z$rf_oof, as.numeric(df_z$Z)-1))
print(cor(df_z$lm_in_sample, as.numeric(df_z$Z)-1))
print(cor(df_z$rf_oof, df_z$lm_in_sample))
cat("RMSE:\n")
print(sqrt(mean((df_z$rf_oof - as.numeric(df_z$Z)+1)^2)))
print(sqrt(mean((df_z$lm_in_sample - as.numeric(df_z$Z)+1)^2)))


### 5) OOF for p(D=1|Z,X) ----
df_d <- data.frame(
  rowIndex=seq_len(n),
  D=factor(ifelse(D==1,"Class2","Class1"), levels=c("Class1","Class2")),
  Z=factor(ifelse(Z==1,"Class2","Class1"), levels=c("Class1","Class2")),
  X1=factor(X1), X2=factor(X2), X3=factor(X3), X4=factor(X4),
  pD0=pD0, pD1=pD1
)
K <- 5
fold_idx <- createFolds(df_d$D, k=K, list=TRUE)

pD1_zC1_glm <- numeric(n)
pD1_zC2_glm <- numeric(n)

pD1_zC1_rf  <- numeric(n)
pD1_zC2_rf  <- numeric(n)

pD1_zC1_xgb <- numeric(n)
pD1_zC2_xgb <- numeric(n)

# tune grid for RF and XGB
rf_grid <- data.frame(mtry = c(2,3))
xgb_grid2 <- expand.grid(
  nrounds          = c(300, 600),
  max_depth        = c(3, 6),
  eta              = c(0.1),
  gamma            = c(0),
  colsample_bytree = c(1),
  min_child_weight = c(1),
  subsample        = c(1)
)
ctrl_d <- trainControl(
  method="cv", number=3,
  classProbs=TRUE, summaryFunction=twoClassSummary,
  verboseIter=FALSE, savePredictions="final"
)

for(k in seq_len(K)){
  test_ids  <- fold_idx[[k]]
  train_ids <- setdiff(seq_len(n), test_ids)
  
  d_train <- df_d[train_ids, ]
  d_test  <- df_d[test_ids, ]
  
  # (A) Logistic
  glm_fit <- train(
    D ~ Z + X1 + X2 + X3 + X4,
    data      = d_train,
    method    = "glm",
    family    = "binomial",
    metric    = "ROC",
    trControl = ctrl_d
  )
  
  # predict for Z=Class1, Z=Class2
  d_testC1 <- d_test %>% mutate(Z=factor("Class1", levels=c("Class1","Class2")))
  d_testC2 <- d_test %>% mutate(Z=factor("Class2", levels=c("Class1","Class2")))
  
  glm_pC1 <- predict(glm_fit, newdata=d_testC1, type="prob")[,"Class2"]
  glm_pC2 <- predict(glm_fit, newdata=d_testC2, type="prob")[,"Class2"]
  
  pD1_zC1_glm[test_ids] <- glm_pC1
  pD1_zC2_glm[test_ids] <- glm_pC2
  
  # (B) Random Forest
  rf_fit <- train(
    D ~ Z + X1 + X2 + X3 + X4,
    data      = d_train,
    method    = "rf",
    metric    = "ROC",
    ntree     = 500,
    tuneGrid  = rf_grid,
    trControl = ctrl_d
  )
  
  rf_pC1 <- predict(rf_fit, newdata=d_testC1, type="prob")[,"Class2"]
  rf_pC2 <- predict(rf_fit, newdata=d_testC2, type="prob")[,"Class2"]
  
  pD1_zC1_rf[test_ids] <- rf_pC1
  pD1_zC2_rf[test_ids] <- rf_pC2
  
  # (C) XGB
  xgb_fit <- train(
    D ~ Z + X1 + X2 + X3 + X4,
    data      = d_train,
    method    = "xgbTree",
    metric    = "ROC",
    tuneGrid  = xgb_grid2,
    trControl = ctrl_d
  )
  
  xgb_pC1 <- predict(xgb_fit, newdata=d_testC1, type="prob")[,"Class2"]
  xgb_pC2 <- predict(xgb_fit, newdata=d_testC2, type="prob")[,"Class2"]
  
  pD1_zC1_xgb[test_ids] <- xgb_pC1
  pD1_zC2_xgb[test_ids] <- xgb_pC2
}

# Attach these OOF predictions back to df_d
df_d$pD1_zC1_glm <- pD1_zC1_glm
df_d$pD1_zC2_glm <- pD1_zC2_glm
df_d$pD1_zC1_rf  <- pD1_zC1_rf
df_d$pD1_zC2_rf  <- pD1_zC2_rf
df_d$pD1_zC1_xgb <- pD1_zC1_xgb
df_d$pD1_zC2_xgb <- pD1_zC2_xgb

# Define p(AT), p(NT), p(CP) for each model, compare to true
df_d$pD0 <- df_d$pD0
df_d$pD1 <- df_d$pD1

df_d$pAT_true <- df_d$pD0
df_d$pNT_true <- 1 - df_d$pD1
df_d$pCP_true <- df_d$pD1 - df_d$pD0

# For each model M in {glm, rf, xgb}:
#   pAT_hat(M) = pD1_zC1_M
#   pNT_hat(M) = 1 - pD1_zC2_M
#   pCP_hat(M) = pD1_zC2_M - pD1_zC1_M

df_d <- df_d %>%
  mutate(
    pAT_hat_glm = pD1_zC1_glm,
    pNT_hat_glm = 1 - pD1_zC2_glm,
    pCP_hat_glm = pD1_zC2_glm - pD1_zC1_glm,
    
    pAT_hat_rf  = pD1_zC1_rf,
    pNT_hat_rf  = 1 - pD1_zC2_rf,
    pCP_hat_rf  = pD1_zC2_rf - pD1_zC1_rf,
    
    pAT_hat_xgb = pD1_zC1_xgb,
    pNT_hat_xgb = 1 - pD1_zC2_xgb,
    pCP_hat_xgb = pD1_zC2_xgb - pD1_zC1_xgb
  )

mse <- function(a,b) mean((a-b)^2)

cat("\n-- MSE Results for each model --\n")

# GLM
cat("GLM: MSE(AT) =", mse(df_d$pAT_hat_glm, df_d$pAT_true), "\n")
cat("GLM: MSE(NT) =", mse(df_d$pNT_hat_glm, df_d$pNT_true), "\n")
cat("GLM: MSE(CP) =", mse(df_d$pCP_hat_glm, df_d$pCP_true), "\n\n")

# RF
cat("RF:  MSE(AT) =", mse(df_d$pAT_hat_rf, df_d$pAT_true), "\n")
cat("RF:  MSE(NT) =", mse(df_d$pNT_hat_rf, df_d$pNT_true), "\n")
cat("RF:  MSE(CP) =", mse(df_d$pCP_hat_rf, df_d$pCP_true), "\n\n")

# XGB
cat("XGB: MSE(AT) =", mse(df_d$pAT_hat_xgb, df_d$pAT_true), "\n")
cat("XGB: MSE(NT) =", mse(df_d$pNT_hat_xgb, df_d$pNT_true), "\n")
cat("XGB: MSE(CP) =", mse(df_d$pCP_hat_xgb, df_d$pCP_true), "\n")

### 6) Weights from OOF EztildeX & Compare to True Weights ----
# 1) Compute the "true" weights for ALL observations, then subset by type
tildeZ_D_mean <- mean((Z - lmZ$fitted.values) * D)

w_AT_true_full <- EztildeX * pD0 / tildeZ_D_mean
w_CP_true_full <- (EztildeX + predict(lmZ)*(1 - pZ)) * (pD1 - pD0) / tildeZ_D_mean
w_NT_true_full <- EztildeX * (1 - pD1) / tildeZ_D_mean

# Only define them for each type
w_AT_true <- w_AT_true_full; w_AT_true[type != "AT"] <- NA
w_CP_true <- w_CP_true_full; w_CP_true[type != "C"]  <- NA
w_NT_true <- w_NT_true_full; w_NT_true[type != "NT"] <- NA

# 2) Build data.frame with true weights
df_weights <- data.frame(
  rowIndex = seq_len(n),
  type     = type,
  w_AT_true = w_AT_true,
  w_CP_true = w_CP_true,
  w_NT_true = w_NT_true
)

# 3) Merge OOF EztildeX_hat & pD0, pD1, plus lm_in_sample if needed
df_weights <- df_weights %>%
  left_join(
    df_z %>% select(rowIndex, EztildeX_hat, lm_in_sample),
    by="rowIndex"
  ) %>%
  left_join(
    df_d %>% select(rowIndex, pD0, pD1),
    by="rowIndex"
  ) %>%
  left_join(
    data.frame(rowIndex=seq_len(n), pZ=pZ),
    by="rowIndex"
  )

# 4) Define "estimated" weights (AT, CP, NT) using OOF EztildeX_hat
df_weights$w_AT_hat <- df_weights$EztildeX_hat * df_weights$pD0 / tildeZ_D_mean
df_weights$w_AT_hat[df_weights$type!="AT"] <- NA

df_weights$w_CP_hat <- (df_weights$EztildeX_hat + df_weights$lm_in_sample*(1 - df_weights$pZ)) *
  (df_weights$pD1 - df_weights$pD0) / tildeZ_D_mean
df_weights$w_CP_hat[df_weights$type!="C"] <- NA

df_weights$w_NT_hat <- df_weights$EztildeX_hat * (1 - df_weights$pD1) / tildeZ_D_mean
df_weights$w_NT_hat[df_weights$type!="NT"] <- NA

# 5) Compare via MSE and summary
mse_w <- function(a,b) mean((a - b)^2, na.rm=TRUE)

cat("MSE(AT weights):", mse_w(df_weights$w_AT_hat, df_weights$w_AT_true), "\n")
cat("MSE(CP weights):", mse_w(df_weights$w_CP_hat, df_weights$w_CP_true), "\n")
cat("MSE(NT weights):", mse_w(df_weights$w_NT_hat, df_weights$w_NT_true), "\n\n")

cat("Summary of AT weight error:\n")
print(summary(df_weights$w_AT_hat - df_weights$w_AT_true))

cat("\nSummary of CP weight error:\n")
print(summary(df_weights$w_CP_hat - df_weights$w_CP_true))

cat("\nSummary of NT weight error:\n")
print(summary(df_weights$w_NT_hat - df_weights$w_NT_true))


# 6) Plot the estimated vs. true weights: gather them into long format
plot_data <- df_weights %>%
  select(
    rowIndex, type,
    w_AT_true, w_AT_hat,
    w_CP_true, w_CP_hat,
    w_NT_true, w_NT_hat
  ) %>%
  pivot_longer(
    cols = starts_with("w_"),
    names_to = c("weight_type", "measure"),
    names_pattern = "w_(AT|CP|NT)_(true|hat)",
    values_to = "value"
  ) %>%
  filter(!is.na(value))

ggplot(plot_data, aes(x=measure, y=value, fill=measure)) +
  geom_boxplot(alpha=0.5) +
  facet_wrap(~weight_type, scales="free_y") +
  theme_bw() +
  labs(
    x="", y="Weight"
  ) +
  scale_fill_manual(values=c("red","blue"),labels=c("True","Estimated")) +
  # no x-axis labels and ticks
  theme(axis.text.x=element_blank(), axis.ticks.x=element_blank())
ggsave("output/sim_weights_boxplot.svg", width=5, height=5)

plot_pairs <- bind_rows(
  # AT
  df_weights %>%
    filter(!is.na(w_AT_true) & !is.na(w_AT_hat)) %>%
    transmute(
      type = "AT",
      x    = w_AT_true,
      y    = w_AT_hat
    ),
  # CP
  df_weights %>%
    filter(!is.na(w_CP_true) & !is.na(w_CP_hat)) %>%
    transmute(
      type = "CP",
      x    = w_CP_true,
      y    = w_CP_hat
    ),
  # NT
  df_weights %>%
    filter(!is.na(w_NT_true) & !is.na(w_NT_hat)) %>%
    transmute(
      type = "NT",
      x    = w_NT_true,
      y    = w_NT_hat
    )
)
ggplot(plot_pairs, aes(x=x, y=y)) +
  geom_point(alpha=0.5) +
  facet_wrap(~type, scales="free") +
  geom_abline(intercept=0, slope=1, color="red", linetype="dashed", lwd=1) +
  labs(
    x     = "True Weight",
    y     = "Estimated Weight"
  ) +
  geom_smooth(method="lm", se=TRUE, color="blue") +
  theme_bw()
ggsave("output/sim_weights_scatter.svg", width=5, height=5)

diff_data <- plot_pairs %>%
  mutate(diff = y - x)

ggplot(diff_data, aes(x=diff)) +
  geom_histogram(bins=50, fill="steelblue", alpha=0.6) +
  facet_wrap(~type, scales="free_y") +
  geom_vline(xintercept=0, color="red", linetype="dashed") +
  labs(
    x="(Estimated - True) Weight",
    y="Count"
  ) +
  theme_bw()
ggsave("output/sim_weights_diff_hist.svg", width=5, height=5)
