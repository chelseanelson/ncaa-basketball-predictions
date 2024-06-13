# Tuning Resamples ----
# Define and fit initial k-nearest neighbors model
# BE AWARE: there is a random process in this script (seed set right before it)

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

# load pre-processing/feature engineering recipe 
load(here("recipes/recipe_nonpara.rda"))

# model specifications ----
knn_model <- 
  nearest_neighbor(
    mode = "classification",
    neighbors = tune()
  ) %>%
  set_engine("kknn")

# define workflows 
knn_wflow <-
  workflow() %>%
  add_model(knn_model) %>%
  add_recipe(recipe_nonpara)

# hyperparameter tuning values ----

# check ranges for hyperparameters 
hardhat::extract_parameter_set_dials(knn_model)

# change hyperparameter ranges 
knn_params <- extract_parameter_set_dials(knn_model)

# build tuning grid 
knn_grid <- grid_regular(knn_params, levels = 5)

# tune workflows/models ----
# set seed
set.seed(1011)
tic.clearlog() # clear log

tic("knn") # start clock

tuned_knn <-
  knn_wflow %>%
  tune_grid(
    cbb_folds,
    grid = knn_grid,
    control = control_grid(save_workflow = TRUE)
  )
toc(log = TRUE)

# Extract runtime info
time_log <- tic.log(format = FALSE)

tictoc_knn <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

# write out results (fitted/trained workflows)
save(tuned_knn,tictoc_knn, file = here("results/tuned_models/tuned_knn.rda"))