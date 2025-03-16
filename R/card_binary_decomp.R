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

set.seed(42)
### 1) Data ----
card <- read_dta("https://raw.github.com/scunning1975/mixtape/master/card.dta")
attach(card)
Y <- lwage
D <- as.numeric(educ > 13)
Z <- nearc4
expbin <- as.numeric(exper > 8)
X <- cbind(expbin, black, south, smsa)
idx_complete <- complete.cases(Y, D, X, Z)
Y <- Y[idx_complete]
D <- D[idx_complete]
Z <- Z[idx_complete]
X <- X[idx_complete, , drop=FALSE]
detach(card)
df <- data.frame(Y=Y, D=D, Z=Z,
                 expbin=X[,"expbin"],
                 black=X[,"black"],
                 south=X[,"south"],
                 smsa=X[,"smsa"])
n <- nrow(df)

# Define factor versions for grouping and modeling
df$expbin_f <- factor(df$expbin, levels=c(0,1), labels=c("exp<=8","exp>8"))
df$black_f  <- factor(df$black,  levels=c(0,1), labels=c("nonblack","black"))
df$south_f  <- factor(df$south,  levels=c(0,1), labels=c("nonsouth","south"))
df$smsa_f   <- factor(df$smsa,   levels=c(0,1), labels=c("nonsmsa","smsa"))
df$Z_factor <- factor(df$Z, levels=c(0,1), labels=c("Class0","Class1"))
df$D_factor <- factor(df$D, levels=c(0,1), labels=c("Class0","Class1"))

### 2) MODEL COMPARISON: E[Z | X] ----

df_z <- data.frame(
  Z = df$Z_factor,
  expbin = df$expbin_f,
  black  = df$black_f,
  south  = df$south_f,
  smsa   = df$smsa_f
)
df_z$rowIndex <- seq_len(n)

ctrl_z <- trainControl(
  method="cv", number=10,
  classProbs=TRUE,
  summaryFunction=twoClassSummary,
  savePredictions="all"
)

set.seed(123)
# (A) Logistic
glm_model <- train(
  Z ~ expbin * black * south * smsa,
  data      = df_z,
  method    = "glm",
  family    = "binomial",
  metric    = "ROC",
  trControl = ctrl_z
)

# (B) Random Forest
rf_grid <- data.frame(mtry=c(2,3,4))
rf_model <- train(
  Z ~ expbin + black + south + smsa,
  data      = df_z,
  method    = "rf",
  metric    = "ROC",
  trControl = ctrl_z,
  ntree     = 1000,
  tuneGrid  = rf_grid
)
# (C) XGBoost
xgb_grid <- expand.grid(
  nrounds          = c(200, 400, 600),
  max_depth        = c(2, 3, 4),
  eta              = c(0.1),
  gamma            = c(0),
  colsample_bytree = c(1),
  min_child_weight = c(1),
  subsample        = c(1)
)
xgb_model <- train(
  Z ~ expbin + black + south + smsa,
  data      = df_z,
  method    = "xgbTree",
  metric    = "ROC",
  trControl = ctrl_z,
  tuneGrid  = xgb_grid
)

# Compare
res_z <- resamples(list(glm=glm_model, rf=rf_model, xgb=xgb_model))
summary(res_z)
models <- c("glm","rf","xgb")
best_model_name <- models[which.max(c(
  glm_model$results[glm_model$results$ROC == max(glm_model$results$ROC),"ROC"],
  rf_model$results[rf_model$results$ROC == max(rf_model$results$ROC),"ROC"],
  xgb_model$results[xgb_model$results$ROC == max(xgb_model$results$ROC),"ROC"]
))]
cat("Best model for Z ~ X is:", best_model_name, "\n")

# Extract OOF predictions from that best model
if(best_model_name=="glm"){
  best_mod <- glm_model
} else if(best_model_name=="rf"){
  best_mod <- rf_model
} else {
  best_mod <- xgb_model
}

best_tune <- best_mod$bestTune
df_oof <- best_mod$pred
for(nm in names(best_tune)){
  df_oof <- df_oof[df_oof[[nm]]==best_tune[[nm]], ]
}
# Average if multiple folds produce duplicates
df_oof_z <- df_oof %>%
  group_by(rowIndex) %>%
  summarize(z_hat_oof = mean(Class1), .groups="drop")

# Merge with df_z
df_z <- left_join(df_z, df_oof_z, by="rowIndex")

# linear model for Z ~ X
lmZ <- lm(Z ~ expbin + black + south + smsa, data=df)
df_z$lm_in_sample <- predict(lmZ)

# ztilde = Z - lmZ; EtildeZhat = z_hat_oof - lm_in_sample
df_z$Z_num <- as.numeric(df_z$Z) - 1
df_z$ztilde <- df_z$Z_num - df_z$lm_in_sample
df_z$EztildeX_hat <- df_z$z_hat_oof - df_z$lm_in_sample
df <- df %>%
  mutate(
    rowIndex = seq_len(n)
  ) %>%
  left_join(
    df_z %>% select(rowIndex, ztilde, EztildeX_hat, lm_in_sample, z_hat_oof),
    by="rowIndex"
  )

### 3) MODEL COMPARISON: p(D=1|Z,X) ----
df_d <- data.frame(
  D = df$D_factor,
  Z = df$Z_factor,
  expbin = df$expbin_f,
  black  = df$black_f,
  south  = df$south_f,
  smsa   = df$smsa_f
)
df_d$rowIndex <- seq_len(n)
ctrl_d <- trainControl(
  method="cv", number=10,
  classProbs=TRUE,
  summaryFunction=twoClassSummary,
  savePredictions="all"
)
set.seed(234)
# (A) Logistic
glmD <- train(
  D ~ Z * expbin * black * south * smsa,
  data      = df_d,
  method    = "glm",
  family    = "binomial",
  metric    = "ROC",
  trControl = ctrl_d
)
# (B) RF
rf_grid2 <- data.frame(mtry=c(2,3,4))
rfD <- train(
  D ~ Z + expbin + black + south + smsa,
  data      = df_d,
  method    = "rf",
  metric    = "ROC",
  ntree     = 1000,
  tuneGrid  = rf_grid2,
  trControl = ctrl_d
)
# (C) XGB
xgb_grid2 <- expand.grid(
  nrounds          = c(200, 400, 600),
  max_depth        = c(2,3,4),
  eta              = c(0.1),
  gamma            = c(0),
  colsample_bytree = c(1),
  min_child_weight = c(1),
  subsample        = c(1)
)
xgbD <- train(
  D ~ Z + expbin + black + south + smsa,
  data      = df_d,
  method    = "xgbTree",
  metric    = "ROC",
  tuneGrid  = xgb_grid2,
  trControl = ctrl_d
)

res_d <- resamples(list(glm=glmD, rf=rfD, xgb=xgbD))
summary(res_d)

modelsp <- c("glm","rf","xgb")
best_d_name <- modelsp[which.max(c(
  max(glmD$results$ROC),
  max(rfD$results$ROC),
  max(xgbD$results$ROC)
))]
cat("Best model for D ~ (Z,X) is:", best_d_name, "\n")

if(best_d_name=="glm"){
  best_d <- glmD
} else if(best_d_name=="rf"){
  best_d <- rfD
} else {
  best_d <- xgbD
}
best_tune_d <- best_d$bestTune
df_oof_d <- best_d$pred
for(nm in names(best_tune_d)){
  df_oof_d <- df_oof_d[df_oof_d[[nm]]==best_tune_d[[nm]],]
}
df_oof_d <- df_oof_d %>%
  group_by(rowIndex) %>%
  summarize(prob_hat = mean(Class1), .groups="drop")

# BUT we also need p(D=1|Z=0,X) and p(D=1|Z=1,X), out-of-fold.

K <- 10
fold_idx <- createFolds(df_d$D, k=K, list=TRUE)
pD1_Z0 <- numeric(n)
pD1_Z1 <- numeric(n)
for(k in seq_len(K)){
  test_ids  <- fold_idx[[k]]
  train_ids <- setdiff(seq_len(n), test_ids)
  
  d_train <- df_d[train_ids,]
  d_test  <- df_d[test_ids,]
  # Fit the chosen method "best_d_name" with the best tuning
  if(best_d_name=="glm"){
    # logistic
    mod_k <- glm(D ~ Z + expbin + black + south + smsa,
                 data=d_train, family=binomial)
  } else if(best_d_name=="rf"){
    mod_k <- randomForest(
      D ~ Z + expbin + black + south + smsa,
      data=d_train, ntree=500, mtry=best_tune_d$mtry
    )
  } else {
    # xgb
    mod_k <- train(
      D ~ Z + expbin + black + south + smsa,
      data=d_train,
      method="xgbTree",
      tuneGrid=best_tune_d,
      metric="ROC",
      trControl=trainControl(method="none", classProbs=TRUE, summaryFunction=twoClassSummary)
    )
  }
  # holdout
  test0 <- d_test; test0$Z <- factor("Class0", levels=c("Class0","Class1"))
  test1 <- d_test; test1$Z <- factor("Class1", levels=c("Class0","Class1"))
  
  if(best_d_name=="glm"){
    p0 <- predict(mod_k, newdata=test0, type="response")
    p1 <- predict(mod_k, newdata=test1, type="response")
  } else if(best_d_name=="rf"){
    p0 <- predict(mod_k, newdata=test0, type="prob")[,"Class1"]
    p1 <- predict(mod_k, newdata=test1, type="prob")[,"Class1"]
  } else {
    p0 <- predict(mod_k, newdata=test0, type="prob")[,"Class1"]
    p1 <- predict(mod_k, newdata=test1, type="prob")[,"Class1"]
  }
  
  pD1_Z0[test_ids] <- p0
  pD1_Z1[test_ids] <- p1
}

df <- df %>%
  mutate(
    pD1_Z0_hat = pD1_Z0,
    pD1_Z1_hat = pD1_Z1
  )

# p(AT|X), p(NT|X), p(CP|X)
df <- df %>%
  mutate(
    pAT_hat = pD1_Z0_hat,
    pNT_hat = 1 - pD1_Z1_hat,
    pCP_hat = pD1_Z1_hat - pD1_Z0_hat
  )

### 4) DEFINE BETA–RICH WEIGHTS w_AT, w_CP, w_NT ----
# The "denominator" is mean( (Z - lmZ) * D ) in-sample
df$Z_minus_lmZ <- df$Z - df$lm_in_sample
tildeZ_D_mean  <- mean(df$Z_minus_lmZ * df$D)
# w_AT = EztildeX_hat * p(AT|X) / mean(tildeZ * D)
df$w_AT <- (df$EztildeX_hat) * (df$pAT_hat) / tildeZ_D_mean
# w_CP = [EztildeX_hat + lm_in_sample*(1 - z_hat_oof)] * [p(CP|X)] / mean(tildeZ * D)
df$w_CP <- (df$EztildeX_hat + df$lm_in_sample*(1 - df$z_hat_oof)) * (df$pCP_hat) / tildeZ_D_mean
# w_NT = EztildeX_hat * p(NT|X) / mean(tildeZ * D)
df$w_NT <- df$EztildeX_hat * df$pNT_hat / tildeZ_D_mean


### 5) AGGREGATE WEIGHTS BY COVARIATE GROUP AND PLOT ----
# Define a group label from (expbin, black, south, smsa)
df$group <- apply(df[,c("expbin_f","black_f","south_f","smsa_f")], 1, function(r){
  paste0(r[1],",",r[2],",",r[3],",",r[4])
})
# Summarize mean weights in each group
data_agg <- df %>%
  group_by(group) %>%
  summarize(
    mean_wAT = mean(w_AT, na.rm=TRUE),
    mean_wCP = mean(w_CP, na.rm=TRUE),
    mean_wNT = mean(w_NT, na.rm=TRUE),
    n=n(),
    .groups="drop"
  )
# For a stacked bar, pivot longer:
data_agg_long <- data_agg %>%
  pivot_longer(cols=c("mean_wAT","mean_wCP","mean_wNT"),
               names_to="type", values_to="mean_weight")
data_agg_long$type <- factor(data_agg_long$type,
                             levels=c("mean_wAT","mean_wCP","mean_wNT"),
                             labels=c("AT","CP","NT"))
# Sort groups by sum of weights or something
data_agg_long <- data_agg_long %>%
  group_by(group) %>%
  mutate(total_weight = sum(mean_weight)) %>%
  ungroup() %>%
  arrange(total_weight)
data_agg_long$group <- factor(data_agg_long$group, levels=unique(data_agg_long$group))
p_weights <- ggplot(data_agg_long, aes(x=group, y=mean_weight, fill=type)) +
  geom_col(position="dodge") +
  coord_flip() +
  labs(x="Covariate Group", y="Mean Beta–Rich Weight", fill="Type") +
  theme_bw() +
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank())
p_weights <- p_weights + 
  geom_text(
    data = data_agg_long %>% filter(type == "NT"),
    aes(label = n),
    position = position_dodge(width = 0.9),
    hjust = -0.1, size = 3
  )
print(p_weights)
ggsave("output/card_weights_by_group.png", p_weights, dpi=300)

data_agg <- df %>%
  group_by(group) %>%
  summarize(
    mean_wAT = mean(w_AT, na.rm=TRUE),  sd_wAT = sd(w_AT, na.rm=TRUE),
    mean_wCP = mean(w_CP, na.rm=TRUE),  sd_wCP = sd(w_CP, na.rm=TRUE),
    mean_wNT = mean(w_NT, na.rm=TRUE),  sd_wNT = sd(w_NT, na.rm=TRUE),
    n=n(),
    .groups  = "drop"
  )
data_means_long <- data_agg %>%
  pivot_longer(
    cols         = c("mean_wAT","mean_wCP","mean_wNT"),
    names_to     = "type_mean",
    values_to    = "mean_weight"
  )
data_sds_long <- data_agg %>%
  pivot_longer(
    cols         = c("sd_wAT","sd_wCP","sd_wNT"),
    names_to     = "type_sd",
    values_to    = "sd_weight"
  )
library(stringr)
data_plot_long <- data_means_long %>%
  mutate(type_sd = str_replace(type_mean, "mean_", "sd_")) %>%
  left_join(
    data_sds_long %>% select(group, type_sd, sd_weight),
    by = c("group","type_sd")
  ) %>%
  mutate(
    type = factor(
      type_mean,
      levels = c("mean_wAT","mean_wCP","mean_wNT"),
      labels = c("AT","CP","NT")
    )
  )
data_plot_long <- data_plot_long %>%
  group_by(group) %>%
  mutate(max_mean_w = max(mean_weight, na.rm=TRUE)) %>%
  ungroup() %>%
  # remove group=exp>8,black,nonsouth,nonsmsa for lack of observations
  filter(group != "exp>8,black,nonsouth,nonsmsa")
data_plot_long$group <- reorder(data_plot_long$group, data_plot_long$max_mean_w)
p_weights_sd <- ggplot(data_plot_long, aes(x=group, y=mean_weight, fill=type)) +
  geom_col(position="dodge") +
  geom_errorbar(
    aes(ymin = mean_weight - sd_weight, ymax = mean_weight + sd_weight),
    width = 0.3,
    position = position_dodge(width=0.9)
  ) +
  coord_flip() +
  labs(x="Covariate Combination", y="Mean ± SD of Weights", fill="Type") +
  theme_bw() +
  theme(
    panel.grid.major.y = element_blank(),
    panel.grid.minor   = element_blank(),
    panel.grid.major.x = element_blank(),
    legend.position    = "top"
  )
n_groups <- length(levels(data_plot_long$group))

# Add horizontal lines at group boundaries: 1.5, 2.5, 3.5, etc.
p_weights_sd <- p_weights_sd +
  geom_vline(
    xintercept = seq(1.5, n_groups - 0.5, 1),
    color="grey90"
  )
p_weights_sd + 
  # Add 'n' once per group, from data_agg (one row per group)
  geom_text(
    data = data_agg %>% filter(group != "exp>8,black,nonsouth,nonsmsa"),
    aes(
      x     = group,
      y     = -9,
      label = paste0("n=", n)
    ),
    color = "black",
    size  = 3,
    hjust = 1,
    inherit.aes = FALSE
  ) +
  coord_flip(clip = "off")
ggsave("output/card_weights_by_group_sd.png", width=8, height=6, dpi=300)

