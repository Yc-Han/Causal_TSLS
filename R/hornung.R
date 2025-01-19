library(AER)
library(haven)
library(tidyverse)
library(ddml)

# Load the data
textiles <- as.data.frame(read_dta("hornung-data-textiles.dta"))

# Define Variables
textiles_cleaned <- textiles %>%
  filter(!is.na(poploss_keyser))

Y <- textiles_cleaned$ln_output1802_all
D <- textiles_cleaned$hugue_1700_pc
Z <- textiles_cleaned$poploss_keyser
X <- textiles_cleaned %>%
  select(ln_workers1802_all, ln_looms1802_all, ln_input1802_all, no_looms,
         textil_1680_dummy, ln_popcity1802, vieh1816_schaf_ganz_veredelt_pc,
         pop1816_prot_pc, no_hugue_possible, imputed_dummy) %>%
  as.matrix()

# bind as data frame and keep only complete cases
data <- cbind(Y, D, Z, X)
data <- as.data.frame(data)
data <- data[complete.cases(data),]
# remove observation 65 due to post hoc warning
data <- data[-c(65,39),]

Y <- data$Y
D <- data$D
Z <- data$Z
X <- as.matrix(data[, -c(1, 2, 3)])

# OLS
lr <- lm(Y ~ D + X)
summary(lr)

# IV without covariate
iv_nocov <- ivreg(Y ~ D | Z)
summary(iv_nocov)

# IV with covariate
iv_cov <- ivreg(Y ~ D + X | Z + X)
summary(iv_cov)

# resettest
regZ <- lm(Z ~ X)
resettest(regZ)

# DDML-PLIV
set.seed(123)
learners_multiple <- list(list(fun = ols),
                          list(fun = mdl_glmnet),
                          list(fun = mdl_ranger),
                          list(fun = mdl_xgboost))

pliv_rich <- ddml_pliv(Y, D, Z, X,
                            learners = learners_multiple,
                            ensemble_type = c('nnls', 'nnls1','ols', 'average'),
                            shortstack = TRUE,
                            sample_folds = 3,
                            cv_folds = 3,
                            silent = FALSE)

summary(pliv_rich)
