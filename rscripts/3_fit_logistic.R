# Fitting Resamples ----
# Define and fit initial logistic regression model 

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(doMC)
library(tictoc)

# handle common conflicts 
tidymodels_prefer()

# set up parallel processing
num_cores <- parallel::detectCores(logical = TRUE)
registerDoMC(cores = num_cores - 1)

# load folded data 
load(here("data/data-splitting/cbb_folds.rda"))

# load pre-procesing/feature engineering recipe 
load(here("recipes/recipe_para.rda"))
load(here("results/tuned_models/metrics_set.rda"))

# model specifications ----
logistic_model <-
  multinom_reg(mode = "classification",
               penalty = 1) %>%
  set_engine("glmnet")

# define workflows ---
logistic_wflow <-
  workflow() %>%
  add_model(logistic_model) %>%
  add_recipe(recipe_para)

# fit workflows/models
tic("logistic") # start clock

keep_wflow <- control_resamples(save_workflow = TRUE)

fit_logistic <-
  fit_resamples(
    logistic_wflow,
    resamples = cbb_folds,
    control = keep_wflow,
    metrics = metrics_set
  )
toc(log = TRUE) # stop clock

# Extract runtime info
time_log <- tic.log(format = FALSE)

tictoc_logistic <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

# write out results (fitted/trained workflows) ----
save(
  fit_logistic,
  tictoc_logistic,
  file = here("results/tuned_models/fit_logistic.rda")
)