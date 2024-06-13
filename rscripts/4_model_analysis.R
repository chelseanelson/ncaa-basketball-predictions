# Model Analysis
# Analysis of tuned and trained models (comparison)
# Main Assessment Metric : Accuracy

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(doMC)

# handle common conflicts 
tidymodels_prefer()

# load in tuned models 
load(here("results/tuned_models/fit_logistic.rda"))
load(here("results/tuned_models/fit_nbayes.rda"))
load(here("results/tuned_models/tuned_nnet.rda"))
load(here("results/tuned_models/tuned_knn.rda"))
# load(here("results/tuned_models/tuned_mars.rda")) -- this one is not working 
load(here("results/tuned_models/tuned_rf.rda"))
load(here("results/tuned_models/tuned_svm_radial.rda"))
load(here("results/tuned_models/tuned_bt.rda"))
load(here("results/tuned_models/tuned_svm_poly.rda"))

# comparing sub-models ----
## Multinomial Regressions
logistic_best <- show_best(fit_logistic, metric = "accuracy")

## Naive Bayes 
nbayes_best <- show_best(fit_nbayes, metric = "accuracy")

## Neutral Networks
nnet_plot <- tuned_nnet %>% autoplot(metric = "accuracy")
nnet_best <- show_best(tuned_nnet, metric = "accuracy")

## K-Nearest Neighbors 
knn_plot <- tuned_knn %>% autoplot(metric = "accuracy") 
knn_best <- show_best(tuned_knn, metric = "accuracy")

## Boosted Trees
bt_plot <- tuned_bt %>% autoplot(metric = "accuracy")
bt_best <- show_best(tuned_bt, metric = "accuracy") 

# Random Forests 
rf_plot <- tuned_rf %>% autoplot(metric = "accuracy") 
rf_best <- show_best(tuned_rf, metric = "accuracy")

## MARS
mars_plot <- tuned_mars %>% autoplot(metric = "accuracy")
mars_best <- show_best(tuned_mars, metric = "accuracy") 

## SVM Polynomial
svm_poly_plot <- tuned_svm_poly %>% autoplot(metric = "accuracy")
svm_poly_best <- show_best(tuned_svm_poly, metric = "accuracy") 

## SVM Radial
svm_radial_plot <- tuned_svm_radial %>% autoplot(metric = "accuracy")
svm_radial_best <- show_best(tuned_svm_radial, metric = "accuracy") 


# Results 
runtime <- bind_rows(tictoc_nbayes,
                     tictoc_logistic,
                     #tictoc_mars,
                     tictoc_nnet,
                     tictoc_svm_radial,
                     tictoc_svm_poly,
                     tictoc_knn,
                     tictoc_rf,
                     tictoc_bt
) %>%
  select(model, runtime)

# select best of each
get_best <- function(tuned_model, name) {
  show_best(tuned_model, metric = "accuracy") %>%
    mutate(model = name) %>%
    slice_head(n = 1) %>%
    select(model, mean, std_err)
}

nbayes <- get_best(fit_nbayes, "nbayes")

logistic <- get_best(fit_logistic, "logistic")

#mars <- get_best(tuned_mars, "MARS")

nnet <- get_best(tuned_nnet, "nnet")

svm_radial <- get_best(tuned_svm_radial, "svm_radial")

svm_poly <- get_best(tuned_svm_poly, "svm_poly")

knn <- get_best(tuned_knn, "knn")

rf <- get_best(tuned_rf, "rf")

bt <- get_best(tuned_bt, "bt")


# combine best and runtime
initial_results <- bind_rows(nbayes,
                             logistic,
                             #mars,
                             nnet,
                             svm_radial,
                             svm_poly,
                             knn,
                             rf,
                             bt
) %>%
  left_join(runtime) %>% arrange(desc(mean))


# Best Model: Random Forest 

# write out results (plots, tables)
write_rds(intital_results, file = here("results/tuned_models/initial_results.rds"))