# Recipes ----
# Creating recipes 

# Load package(s)
library(tidymodels)
library(tidyverse)
library(here)

# handle common conflicts
tidymodels_prefer()

# load training data 
load(here("data/data-splitting/cbb_train.rda"))

# Featured Engineered Recipes 1 ----

## recipe 1 (naive bayes) ----
recipe_naivebayes <- recipe(postseason ~., data = cbb_train) %>% 
  step_novel(all_nominal_predictors()) %>% 
  step_nzv(all_numeric_predictors()) %>% 
  step_normalize(all_numeric_predictors())

# check recipe
recipe_naivebayes %>% 
  prep() %>% 
  bake(new_data = NULL) %>%
  glimpse()

## recipe 2 (parametric) ----

recipe_para <- recipe(postseason ~., data = cbb_train) %>% 
  step_novel(all_nominal_predictors()) %>% 
  step_corr(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>% 
  step_nzv(all_predictors()) %>% 
  step_normalize(all_predictors())

# check recipe
recipe_para %>% 
  prep() %>% 
  bake(new_data = NULL) %>% 
  glimpse()

## recipe 3 (nonparametric) ----

recipe_nonpara <- recipe(postseason ~., data = cbb_train) %>%
  step_novel(all_nominal_predictors()) %>%
  step_corr(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_nzv(all_predictors()) %>%
  step_normalize(all_predictors())

#check recipe
recipe_nonpara %>% 
  prep() %>%
  bake(new_data = NULL) %>%
  glimpse()

# Feature Engineered Recipes 2 ----

## recipe 2 (parametric) ----

recipe_para_2 <- recipe(is_canceled ~., data = hotel_train) %>% 
  step_novel(all_nominal_predictors()) %>% 
  step_other(all_nominal_predictors(), threshold = 0.05) %>%
  step_corr(all_numeric_predictors()) %>%
  step_YeoJohnson(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>% 
  step_nzv(all_predictors()) %>% 
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), num_comp = tune())

# check recipe
# recipe_para_2 %>% 
#   prep() %>% 
#   bake(new_data = NULL) %>% 
#   glimpse()

## recipe 3 (nonparametric) ----

recipe_nonpara_2 <- recipe(is_canceled ~., data = hotel_train) %>%
  step_novel(all_nominal_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.05) %>% 
  step_corr(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_nzv(all_predictors()) %>%
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), num_comp = tune())

# check recipe
# recipe_nonpara_2 %>% 
#   prep() %>% 
#   bake(new_data = NULL) %>% 
#   glimpse()

# metrics set

metrics_set <- metric_set(accuracy, mcc, roc_auc)

# save recipes ----
save(recipe_naivebayes, file = here("recipes/recipe_naivebayes.rda"))

save(recipe_para, file = here("recipes/recipe_para.rda"))

save(recipe_nonpara, 
     file = here("recipes/recipe_nonpara.rda"))


save(recipe_para_2, file = here("recipes/recipe_para_2.rda"))

save(recipe_nonpara_2, 
     file = here("recipes/recipe_nonpara_2.rda"))

save(metrics_set, 
     file = here("results/tuned_models/metrics_set.rda"))
