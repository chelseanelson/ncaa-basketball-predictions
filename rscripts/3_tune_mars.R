# Tuning Resamples ---- # this one keeps messing up 
# Define and fit initial MARS model
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
load(here("recipes/recipe_para.rda"))
load(here("results/tuned_models/metrics_set.rda"))

# model specifications ----
mars_model <-
  mars(
    mode = "classification",
    num_terms = tune(),
    prod_degree = tune()
  ) %>%
  set_engine("earth")

# define workflows
mars_wflow <-
  workflow() %>%
  add_model(mars_model) %>%
  add_recipe(recipe_para)

# hyperparameter tuning values ----

# check ranges for hyperparameters
hardhat::extract_parameter_set_dials(mars_model)

# change hyperparameter ranges
mars_params <- extract_parameter_set_dials(mars_model)

# build tuning grid
mars_grid <- grid_regular(mars_params, levels = 5)

# tune workflows/models ----
# set seed
set.seed(203)

tic("mars")

tuned_mars <-
  mars_wflow %>%
  tune_grid(
    cbb_folds,
    grid = mars_grid,
    control = control_grid(save_workflow = TRUE),
    metrics = metrics_set
  )

toc(log = TRUE)
time_log <- tic.log(format = FALSE)

tictoc_mars <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

# write out results (fitted/trained workflows)
save(tuned_mars, tictoc_mars,
     file = here("results/tuned_models/tuned_mars.rda"))
