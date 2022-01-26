library(tidytext)
library(data.table)
library(tidyverse)
library(tidymodels)

read.csv("/Users/a12517/python_1/all readability.csv") -> raw_d

train %>% # this train coming from create data set..
  janitor::clean_names() %>% 
  tibble -> train

multicomplaints_split <- initial_split(train, 
                                       strata = discourse_type)

multicomplaints_train <- training(multicomplaints_split)
multicomplaints_test <- testing(multicomplaints_split)


multicomplaints_train %>% 
  group_by(discourse_type) %>% 
  mutate(wanted = mean(hapax*10000),
         wanted = round(wanted, -2)) %>% 
  ungroup() -> multicomplaints_train

library(tidylo)

multicomplaints_train %>% 
  select(wanted,
         discourse_text) %>% 
  rowid_to_column() %>% 
  unnest_tokens(word, 
                discourse_text) %>% 
  #distinct(word, .keep_all = TRUE) 
  count(wanted,
        word, 
        sort = TRUE) %>% 
  bind_log_odds(wanted, word, n) %>% 
  group_by(wanted) %>% 
  arrange(-n) %>% 
  slice_head(n = 30) %>% 
  select(wanted,
         word,
         log_odds_weighted) %>% 
  ungroup() -> wanted_words


wanted_words %>% 
  tibble

names <- c(as.character(1:100))

multicomplaints_train %>% 
  group_by(hapax) %>% 
  tally

#join with data
multicomplaints_train %>% 
  as_tibble() %>% 
  select(hapax,
         discourse_text) %>% 
  rowid_to_column() %>% 
  unnest_tokens(word, 
                discourse_text) %>% 
  right_join(weighted_words %>% 
               select(hapax,
                      word,
                      log_odds_weighted), by =c("hapax", "word")) %>% 
  tibble %>% 
  group_by(rowid) %>% 
  summarize(discourse_small= str_c(log_odds_weighted, collapse = " ")) %>% 
  separate(discourse_small, c(names), " ") %>% 
  mutate(across(where(is.character), as.numeric)) %>% 
  replace(is.na(.), 0) -> new_matrix


new_matrix %>% 
  janitor::clean_names() %>% 
  tibble -> new_matrix

multicomplaints_train %>% 
  rowid_to_column() %>% 
  left_join(new_matrix, by = c("rowid")) %>% 
  tibble -> new_raw

new_raw[3:50,] %>% View

#------------
colnames(new_raw)

new_raw %>% 
  select(discourse_text, 
         discourse_type, 
         TTR, 
         colman_grade, 
         hapax, 
         wanted,
         23:197) %>% 
  tibble -> new_raw_first

train <- new_raw_first

#----------------
multicomplaints_test %>% 
  mutate(wanted = hapax*10000,
         wanted = round(wanted, -2)) %>% 
  ungroup() -> multicomplaints_test

multicomplaints_test %>% 
  group_by(wanted) %>% 
  tally

multicomplaints_test %>% 
  select(discourse_text, 
         discourse_type, 
         TTR, 
         colman_grade, 
         hapax, 
         wanted) -> multicomplaints_test

multicomplaints_test %>% 
  mutate(
    wanted = case_when(
      wanted <= 5400 ~ "5400",
      wanted <= 5700 ~ "5700",
      wanted <= 6100 ~ "6100",
      wanted <= 7500 ~ "7500",
      wanted <= 8100 ~ "8100",
      wanted <= 8800 ~ "8800",
      TRUE ~ "8900"
    )
  ) %>% 
  tibble -> multicomplaints_test

multicomplaints_test %>% 
  glimpse

wanted_words %>% 
  group_by(wanted) %>% 
  tally

multicomplaints_test %>% 
  as_tibble() %>% 
  select(hapax,
         discourse_text) %>% 
  rowid_to_column() %>% 
  unnest_tokens(word, 
                discourse_text) %>%  
  left_join(weighted_words %>% 
               select(hapax,
                      word,
                      log_odds_weighted), by =c("hapax", "word")) %>% 
  tibble %>% 
  group_by(rowid) %>% 
  summarize(discourse_small= str_c(log_odds_weighted, collapse = " ")) %>% 
  separate(discourse_small, c(names), " ") %>% 
  mutate(across(where(is.character), as.numeric)) %>% 
  replace(is.na(.), 0) -> new_log


test_log %>% 
  glimpse

test_log %>% 
  janitor::clean_names() %>% 
  tibble -> test_log

multicomplaints_test %>% 
  rowid_to_column() %>% 
  left_join(new_matrix, by = c("rowid")) %>% 
  tibble -> new_test

colnames(new_test)


test <- new_test
#------
train %>% 
  replace(is.na(.), 0) -> train

test %>% 
  replace(is.na(.), 0) -> test

train %>% 
  write.csv("train.csv")

test %>% 
  write.csv("test.csv")
#------------------
library(themis)
library(textrecipes)

read.csv("train.csv") -> train
read.csv("test.csv") -> test

train %>% 
  glimpse

test %>% 
  glimpse

train %>% 
  replace(is.na(.), 0) -> train

test %>% 
  replace(is.na(.), 0) -> test

train %>% 
  select(-rowid) -> train

test %>% 
  select(-rowid) -> test


multicomplaints_rec <-
  recipe(discourse_type ~ .
         ,
         data = train) %>% 
  step_tokenize(discourse_text) %>%
  step_tokenfilter(discourse_text, max_tokens = 300) %>%
  textrecipes::step_tfidf(discourse_text) %>% 
  themis::step_downsample(discourse_type)

multicomplaints_folds <- vfold_cv(train, 
                                  5)

multi_spec <- multinom_reg(penalty = tune(), 
                           mixture = 1) %>%
  set_mode("classification") %>%
  set_engine("glmnet")

multi_spec

library(hardhat)
sparse_bp <- default_recipe_blueprint(composition = "dgCMatrix")

multi_lasso_wf <- workflow() %>%
  add_recipe(multicomplaints_rec, 
             blueprint = sparse_bp) %>%
  add_model(multi_spec)

multi_lasso_wf

smaller_lambda <- grid_regular(penalty(range = c(-5, 0)), levels = 10)
smaller_lambda

multi_lasso_rs <- tune_grid(
  multi_lasso_wf,
  multicomplaints_folds,
  grid = smaller_lambda,
  control = control_resamples(save_pred = TRUE)
)

best_acc <- multi_lasso_rs %>%
  show_best("accuracy")

best_acc

multi_lasso_rs %>%
  collect_predictions() %>% 
  #f_meas(discourse_type, .pred_class)
  filter(penalty == best_acc$penalty) %>% 
  filter(id == "Fold1") %>% 
  conf_mat(discourse_type, .pred_class) %>% 
  autoplot(type = "heatmap") +
  scale_y_discrete(labels = function(x) str_wrap(x, 20)) +
  scale_x_discrete(labels = function(x) str_wrap(x, 20))

collect_metrics(multi_lasso_rs)
autoplot(multi_lasso_rs)

multi_lasso_rs %>%
  show_best("roc_auc")

best_roc_auc <- select_best(multi_lasso_rs, "roc_auc")
best_roc_auc

collect_predictions(multi_lasso_rs, 
                    parameters = best_roc_auc)

collect_predictions(multi_lasso_rs, 
                    parameters = best_roc_auc) %>% 
  conf_mat(discourse_type, .pred_class) %>% 
  autoplot(type = "heatmap") +
  scale_y_discrete(labels = function(x) str_wrap(x, 20)) +
  scale_x_discrete(labels = function(x) str_wrap(x, 20))

wf_spec_final <- finalize_workflow(multi_lasso_wf, 
                                   best_roc_auc)

library(vip)

wf_spec_final %>%
  fit(train) %>%
  pull_workflow_fit() %>%
  vi(lambda = best_roc_auc$penalty) %>%
  filter(!str_detect(Variable, "tfidf")) %>%
  filter(Importance != 0)


vi_data <- wf_spec_final %>%
  fit(train) %>%
  pull_workflow_fit() %>%
  vi(lambda = best_roc_auc$penalty) %>%
  mutate(Variable = str_remove_all(Variable, "tfidf_discourse_text_")) %>%
  filter(Importance != 0)

vi_data %>% 
  View
#------
test %>% 
  mutate(wanted = as.double(wanted)) -> test

car_wflow_fit <- wf_spec_final %>% 
  fit(data = train) %>% 
  predict(new_data = test)

dim(test)

test %>%
  tibble %>% 
  bind_cols(car_wflow_fit) -> new


new %>% 
  conf_mat(discourse_type, .pred_class) %>% 
  autoplot(type = "heatmap") +
  scale_y_discrete(labels = function(x) str_wrap(x, 20)) +
  scale_x_discrete(labels = function(x) str_wrap(x, 20))


#-------
mod <- boost_tree(trees = 500, 
                  mtry = 6, 
                  min_n = 10,
                  tree_depth = 5) %>%
  set_engine("xgboost", eval_metric = 'mlogloss') %>%
  set_mode("classification")

xgboost_wflow <- workflow() %>%
  add_recipe(multicomplaints_rec) %>%
  add_model(mod) %>% 
  fit(train)

car_wflow_fit <- xgboost_wflow %>% 
  fit(data = train) %>% 
  predict(new_data = test)

test %>%
  tibble %>% 
  bind_cols(car_wflow_fit) -> new


new %>% 
  conf_mat(discourse_type, .pred_class) %>% 
  autoplot(type = "heatmap") +
  scale_y_discrete(labels = function(x) str_wrap(x, 20)) +
  scale_x_discrete(labels = function(x) str_wrap(x, 20))
