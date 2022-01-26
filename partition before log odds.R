library(tidytext)
library(data.table)
library(tidyverse)
library(tidymodels)

read.csv("/Users/a12517/python_1/all readability.csv") -> raw_d

multicomplaints_split <- initial_split(raw_d, 
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
  slice_head(n = 25) %>% 
  select(wanted,
         word,
         log_odds_weighted) %>% 
  ungroup() -> wanted_words


wanted_words %>% 
  tibble

names <- c(as.character(1:175))

#join with data
multicomplaints_train %>% 
  as_tibble() %>% 
  select(wanted,
         discourse_text) %>% 
  rowid_to_column() %>% 
  unnest_tokens(word, 
                discourse_text) %>% 
  inner_join(wanted_words %>% 
               select(wanted,
                      word,
                      log_odds_weighted), by =c("wanted", "word")) %>% 
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
  left_join(new_matrix, by = c("X" ="rowid")) %>% 
  tibble -> new_raw

new_raw[3:10,] %>% View

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
  mutate(wanted = mean(hapax*10000),
         wanted = round(wanted, -2)) %>% 
  ungroup() -> multicomplaints_test

  
multicomplaints_test %>% 
  as_tibble() %>% 
  select(wanted,
         discourse_text) %>% 
  rowid_to_column() %>% 
  unnest_tokens(word, 
                discourse_text) %>% 
  inner_join(wanted_words %>% 
               select(wanted,
                      word,
                      log_odds_weighted), by =c("wanted", "word")) %>% 
  tibble %>% 
  group_by(rowid) %>% 
  summarize(discourse_small= str_c(log_odds_weighted, collapse = " ")) %>% 
  separate(discourse_small, c(names), " ") %>% 
  mutate(across(where(is.character), as.numeric)) %>% 
  replace(is.na(.), 0) -> test_log


test_log %>% 
  glimpse

test_log %>% 
  janitor::clean_names() %>% 
  tibble -> test_log

multicomplaints_test %>% 
  left_join(new_matrix, by = c("X" ="rowid")) %>% 
  tibble -> new_test

colnames(new_test)

new_test %>% 
  select(discourse_text, 
         discourse_type, 
         TTR, 
         colman_grade, 
         hapax, 
         wanted,
         23:197) %>% 
  tibble -> newest_test

test <- newest_test
#------
train %>% 
  replace(is.na(.), 0) -> train

train %>% 
  write.csv("train.csv")
read.csv("train.csv") -> train


test %>% 
  replace(is.na(.), 0) %>% 
  write.csv("test.csv")
read.csv("test.csv") -> test

#------------------
library(themis)
library(textrecipes)

train %>% 
  glimpse

test %>% 
  glimpse

multicomplaints_rec <-
  recipe(discourse_type ~ .
         ,
         data = train) %>% 
  step_tokenize(discourse_text) %>%
  step_tokenfilter(discourse_text, max_tokens = 300) %>%
  textrecipes::step_tfidf(discourse_text) %>% 
  step_downsample(discourse_type)

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
  fit(multicomplaints_train) %>%
  pull_workflow_fit() %>%
  vi(lambda = best_roc_auc$penalty) %>%
  filter(!str_detect(Variable, "tfidf")) %>%
  filter(Importance != 0)


vi_data <- wf_spec_final %>%
  fit(multicomplaints_train) %>%
  pull_workflow_fit() %>%
  vi(lambda = best_roc_auc$penalty) %>%
  mutate(Variable = str_remove_all(Variable, "tfidf_discourse_text_")) %>%
  filter(Importance != 0)

vi_data %>% 
  View
#------
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
