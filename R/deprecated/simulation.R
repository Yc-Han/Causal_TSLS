library(AER)
library(lmtest)

set.seed(42)

n <- 3000

# 1. Generate covariates X
# Example: 4 continuous covariates
X1_cont <- rnorm(n, mean = 0, sd = 1)
X2_cont <- rnorm(n, mean = 0, sd = 1)
X3_cont <- rnorm(n, mean = 0, sd = 1)
X4_cont <- rnorm(n, mean = 0, sd = 1)
# transform into categorical variables by quantile
X1 <- cut(X1_cont, breaks = quantile(X1_cont, probs = seq(0, 1, 0.25)), include.lowest = TRUE,
          labels = c("Q1","Q2","Q3","Q4"))
X2 <- cut(X2_cont, breaks = quantile(X2_cont, probs = seq(0, 1, 0.25)), include.lowest = TRUE,
          labels = c("Q1","Q2","Q3","Q4"))
X3 <- cut(X3_cont, breaks = quantile(X3_cont, probs = seq(0, 1, 0.25)), include.lowest = TRUE,
          labels = c("Q1","Q2","Q3","Q4"))
X4 <- cut(X4_cont, breaks = quantile(X4_cont, probs = seq(0, 1, 0.25)), include.lowest = TRUE,
          labels = c("Q1","Q2","Q3","Q4"))
# as numeric for calculation
X1 <- as.numeric(X1)
X2 <- as.numeric(X2)
X3 <- as.numeric(X3)
X4 <- as.numeric(X4)
# add an hidden unobserved confounder
X_conf <- rnorm(n, 0, 3)
# 2. Generate instrument Z
# Now Z is a Bernoulli random variable with probability pZ,
logitZ <- -1 + 0.5 * (X1_num * X2_num) +
  0.5 * X3_num - 0.3 * X4_num + 0.3 * (X3_num * X4_num) -
  0.1 * (X1_num * X2_num * X3_num * X4_num) + X_conf - 0.1*X_conf^2 + 0.1*X_conf^3
pZ <- plogis(logitZ)
Z <- rbinom(n, size = 1, prob = pZ)
mean(Z)
lmZ <- lm(Z ~ factor(X1) + factor(X2) + factor(X3) + factor(X4))
summary(lmZ)
resettest(lmZ) # type = regressor does not work if X is dummy
# residual and conditional mean
ztilde <- Z - predict(lmZ)
EztildeX <- pZ - predict(lmZ) # E[tildeZ|X]

# 3. Determine potential and actual treatment D
## sample another confounder used in D() and Y()
X_conf2 <- rnorm(n, 1, 2)
pD0 <- plogis(
  -2.0 - 0.3*(X1_num) + 0.4*(X1_num*X2_num) * X_conf2
  + 0.3*(X3_num*X4_num) * X_conf2
  - 0.1*(X1_num*X3_num*X4_num) + X_conf2 - 0.1*X_conf2^2 + 0.01*X_conf2^3
)
D0 <- rbinom(n, size=1, prob=pD0)
mean(D0)
# Probability that D(1)=1 given X
pD1 <- plogis(
  -1  # bigger intercept 
  + 0.6*(X1_num) + 0.2*(X2_num) + 0.2*(X1_num*X2_num) * X_conf2
  + 0.1*(X3_num*X4_num) * X_conf2
  + 0.1*(X1_num*X2_num*X4_num) - X_conf2 + 0.1*X_conf2^2 - 0.01*X_conf2^3
)
D1 <- rbinom(n, size=1, prob=pD1)
mean(D1)
pD1 <- pmax(pD1, pD0)
U <- runif(n)
D0 <- as.numeric(U < pD0)
D1 <- as.numeric(U < pD1)

D <- ifelse(Z==1, D1, D0)

# 4. Label each subject's type (NT, AT, C, Df) from [D(0), D(1)]:
type <- ifelse(D0==0 & D1==0, "NT",   # never-taker
               ifelse(D0==1 & D1==1, "AT",   # always-taker
                      ifelse(D0==0 & D1==1, "C",    # complier
                             "Df")))         # defier
table(type)
# exogeneity check: ztilde independent of G (type) given X
lmZ_type <- lm(Z ~ factor(X1) + factor(X2) + factor(X3) + factor(X4) + type)
summary(lmZ_type)

# 5. Generate outcomes
# Y(0) as some function of X
Y0 <- 1 + 0.2 * (X1 * X2 * X3 - X4 - X1*X4 + X2*X3 - X2*X4 + X3*X4) +
  # a polynomial term of X1+X2+X3+X4
  0.2 * (X1 + X2 + X3 + X4) - 0.1*(X1 + X2 + X3 + X4)^2 + 0.01*(X1 + X2 + X3 + X4)^3 +
  rnorm(n, mean = 0, sd = 3) + 1*X_conf2 - 0.1*X_conf2^3
mean(Y0)
# Y(1) as some function of X
Y1 <- 2 + 0.3 * (X1 * X2 * X3 + X4 + X1*X2 + X1*X3 + X2*X3 + X2*X4) +
  0.2 * (X1 + X2 + X3 + X4) - 0.1*(X1 + X2 + X3 + X4)^2 + 0.01*(X1 + X2 + X3 + X4)^3 +
  rnorm(n, mean = 0, sd = 3) + 2*X_conf2 + 0.1*X_conf2^3
mean(Y1)
# Y as a function of Y(0), Y(1), and D
Y <- D*Y1 + (1-D)*Y0

# LATE = mean(Y1 - Y0 | CP)
true_late <- mean(Y1[type=="C"] - Y0[type=="C"])
hetero <- sd(Y1[type=="C"] - Y0[type=="C"])
true_late
# D = 28.49883
#################################
# OLS
lm <- lm(Y ~ factor(D) + factor(X1) + factor(X2) + factor(X3) + factor(X4))
summary(lm)
# D = 21.529 (1.156)
# Fit an IV regression
iv_model <- ivreg(Y ~ factor(D) + factor(X1) + factor(X2) + factor(X3) + factor(X4) | factor(Z) + factor(X1) + factor(X2) + factor(X3) + factor(X4))
summary(iv_model, diagnostics = TRUE)
# D = 37.285 (2.450)
# Saturate in X (include all interactions)
iv_model_saturated <- ivreg(
  Y ~ factor(D) + factor(X1)*factor(X2)*factor(X3)*factor(X4)
  | factor(Z) + factor(X1)*factor(X2)*factor(X3)*factor(X4)
)
summary(iv_model_saturated, diagnostics = TRUE)
# D = 27.2186 (1.2701)
# Saturate and Weight
iv_model_sw <- ivreg(
  Y ~ factor(D) + factor(X1)*factor(X2)*factor(X3)*factor(X4)
  | factor(Z)*factor(X1)*factor(X2)*factor(X3)*factor(X4)
)
summary(iv_model_sw, diagnostics = TRUE)
# DDML
library(ddml)
X <- cbind(factor(X1), factor(X2), factor(X3), factor(X4))
learners_multiple <- list(list(fun = ols),
                          list(fun = mdl_glmnet,
                               args= list(alpha = 0.5)),
                          list(fun = mdl_xgboost,
                               args = list(nrounds = 500,
                                           max_depth = 4)),
                          list(fun = mdl_ranger,
                               args = list(num.trees = 500,
                                           mtry = 3)))
pliv <- ddml_pliv(Y, D, Z, X,
                   learners = learners_multiple,
                   ensemble_type = c('nnls', 'ols','average'),
                   shortstack = TRUE,
                   sample_folds = 5,
                   cv_folds = 5,
                   silent = FALSE)
summary(pliv)
# 27.6506 (1.476)
late <- ddml_late(Y, D, Z, X_design,
                  learners = learners_multiple,
                  ensemble_type = c('nnls', 'ols' , 'average'),
                  shortstack = TRUE,
                  sample_folds = 5,
                  cv_folds = 5,
                  silent = FALSE)
summary(late)
# extract and compare all Ds
results <- c(true_late, coef(lm)[2], coef(iv_model)[2], coef(iv_model_saturated)[2], 
  coef(iv_model_sw)[2], pliv$coef[1], late$late[3])
# print with name
names(results) <- c("True LATE", "OLS", "IV", "IV Saturated", "IV Saturated Weighted", "DDML", "DDML LATE")
results
###### dissecting the weights under beta_rich
# w(AT) = EtildezX * pD0 * mean(tildez*D)^-1
w_at <- EztildeX * pD0 * mean(ztilde * D)^-1
w_at <- w_at[type=="AT"]
# w(CP) = (EtildezX + predict(lmZ)*(1-pZ)) * (pD1-pD0) * mean(tildez*D)^-1
w_cp <- (EztildeX + predict(lmZ)*(1-pZ)) * (pD1-pD0) * mean(ztilde * D)^-1
w_cp <- w_cp[type=="C"]
# w(NT) = EtildezX * (1-pD1) * mean(tildez*D)^-1
w_nt <- EztildeX * (1-pD1) * mean(ztilde * D)^-1
w_nt <- w_nt[type=="NT"]

## visualize the weights
library(ggplot2)
library(dplyr)
weights <- data.frame(type = c(rep("AT", length(w_at)),
                                rep("C", length(w_cp)),
                                rep("NT", length(w_nt))),
                      weight = c(w_at, w_cp, w_nt))
ggplot(weights, aes(x = weight, fill = type)) +
  geom_density(alpha = 0.5) +
  theme_bw() +
  xlab("Weights") +
  ylab("Density")
# make a dataframe comprosing Y Z D X type and weights
weights_full <- rep(NA, n)
weights_full[type == "AT"] <- w_at
weights_full[type == "C"]  <- w_cp
weights_full[type == "NT"] <- w_nt
data <- data.frame(Y = Y, Z = Z, D = D, X1 = factor(X1), X2 = factor(X2),
                   X3 = factor(X3), X4 = factor(X4),
                   type = type, weights = weights_full)
# plot the weights by covariates
data$group <- apply(data[, c("X1","X2","X3","X4")], 1, function(r) {
  grp <- character(0)
  if(r[1] == 1) grp <- c(grp, "X1:Q1")
  if(r[2] == 1) grp <- c(grp, "X2:Q1")
  if(r[3] == 1) grp <- c(grp, "X3:Q1")
  if(r[4] == 1) grp <- c(grp, "X4:Q1")
  if(r[1] == 2) grp <- c(grp, "X1:Q2")
  if(r[2] == 2) grp <- c(grp, "X2:Q2")
  if(r[3] == 2) grp <- c(grp, "X3:Q2")
  if(r[4] == 2) grp <- c(grp, "X4:Q2")
  if(r[1] == 3) grp <- c(grp, "X1:Q3")
  if(r[2] == 3) grp <- c(grp, "X2:Q3")
  if(r[3] == 3) grp <- c(grp, "X3:Q3")
  if(r[4] == 3) grp <- c(grp, "X4:Q3")
  if(r[1] == 4) grp <- c(grp, "X1:Q4")
  if(r[2] == 4) grp <- c(grp, "X2:Q4")
  if(r[3] == 4) grp <- c(grp, "X3:Q4")
  if(r[4] == 4) grp <- c(grp, "X4:Q4")
  if(length(grp) == 0) grp <- "None"
  paste(grp, collapse = ",")
})
data$group <- factor(data$group)

data_agg <- data %>%
  group_by(group, type) %>%
  summarize(mean_weight = mean(weights, na.rm = TRUE), .groups = "drop") %>%
  mutate(rank = rank(mean_weight))
data_agg <- data_agg[order(data_agg$rank),]
data_agg$group <- factor(data_agg$group, levels = unique(data_agg$group))
ggplot(data_agg, aes(y = group, x = mean_weight, fill = type)) +
  geom_col(position = "stack") +
  labs(y = "Covaraiate Combination", x = "Mean Weight", fill = "Treatment Type") +
  theme_bw() +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.ticks.y = element_blank(),
        axis.text.y = element_blank(),  # Remove vertical axis labels
        axis.text.x = element_text(size = 8))
        
