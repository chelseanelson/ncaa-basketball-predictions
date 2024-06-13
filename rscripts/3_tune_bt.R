# Tuning Resamples ----
# Define, fit and tune initial boosted trees model 
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

# load pre-procesing/feature engineering recipe 
load(here("recipes/recipe_nonpara.rda"))
load(here("results/tuned_models/metrics_set.rda"))

# model specifications ----
bt_model <- 
  boost_tree(
    mode = "classification",
    mtry = tune(),
    min_n = tune(),
    learn_rate = tune(),
    trees = tune()
  ) %>% 
  set_engine("xgboost")

# define workflows ---
bt_wflow <-
  workflow() %>% 
  add_model(bt_model) %>%
  add_recipe(recipe_nonpara)

# hyperparameter tuning values ----

# check ranges for hyperparameters 
hardhat::extract_parameter_set_dials(bt_model)

# change hyperparameter ranges 
bt_params <- extract_parameter_set_dials(bt_model) %>%
  update(mtry = mtry(c(1,20)),
         learn_rate = learn_rate(range = c(-5, -0.2)),
         trees = trees(range = c(500, 1000))) 

# build tuning grid 
bt_grid <- grid_regular(bt_params, 
                        levels = 5)

# tune workflows/models ----
# set seed 
set.seed(1353)
tic.clearlog() # clear log
tic("bt") # start clock

tuned_bt <-
  bt_wflow %>%
  tune_grid(
    cbb_folds,
    grid = bt_grid,
    control = control_grid(save_workflow = TRUE),
    metrics = metrics_set
  )

toc(log = TRUE) # stop clock

# Extract runtime info
time_log <- tic.log(format = FALSE)

tictoc_bt <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

# write out results (fitted/trained workflows) ----
save(tuned_bt, 
     tictoc_bt, 
     file = here("results/tuned_models/tuned_bt.rda"))