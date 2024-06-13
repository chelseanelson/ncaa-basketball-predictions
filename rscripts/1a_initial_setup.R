# Initial Set Up ----
# Processing training, creating resamples, missingness & initial EDA
# BE AWARE: there are random processes in this script (seed set right before them)

# Load package(s)
library(tidymodels)
library(tidyverse)
library(here)
library(naniar)

# handle common conflicts
tidymodels_prefer()


# load raw data
cbb <- read_csv(here("data/cbb.csv")) %>% janitor::clean_names()

# skim, check missingness
skimr::skim_without_charts(cbb)

# need to change NAs in postseason and seed to represent teams that did not make the postseason at all, rather than being unknowns?

# graph target var (`postseason`)
postseason_distribution_plot <- cbb %>%
  ggplot(aes(postseason)) +
  geom_bar() + theme_minimal() + labs(
    title = "Distribution of Final Postseason Position",
    x = "Postseason Positions",
    y = "Count"
  ) +  geom_text(stat = 'count', aes(label = after_stat(count)), vjust = -1) + theme_minimal()

# mutating to fix NAs

cbb <- cbb %>% 
      mutate(postseason = if_else(is.na(postseason), "Didn't Make", postseason),
             postseason = fct_recode(postseason, "Didn't Make" = "N/A"),
             seed = if_else(is.na(seed), "Didn't Make", seed),
             seed = fct_recode(seed, "Didn't Make" = "N/A")
             )

cbb %>% skimr::skim_without_charts(postseason)
cbb %>% skimr::skim_without_charts(seed)

# set seed
set.seed(3342)

# initial split
cbb_split <- cbb %>%
  initial_split(prop = 0.75, strata = postseason)

cbb_train <- training(cbb_split)

cbb_test <- testing(cbb_split)

# set seed 
set.seed(33422)

# resamples
cbb_folds <- cbb_train |>
  vfold_cv(v = 10, repeats = 5, strata = postseason)

# save splits and updated data
save(
  cbb_train,
  file = here("data/data-splitting/cbb_train.rda")
)

save(
  cbb_test,
  file = here("data/data-splitting/cbb_test.rda")
)

save(
  cbb_folds,
  file = here("data/data-splitting/cbb_folds.rda")
)

write_rds(
  cbb,
  file = here("data/cbb_updated.rds")
)

# save plots 
ggsave(postseason_distribution_plot, file = here("figures/postseason_distribution_plot.png"))
