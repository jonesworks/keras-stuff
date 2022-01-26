library(tidytext)
library(quanteda)
library(quanteda.textstats) 
library(data.table)
library(koRpus)
library(tidyverse)

train <- fread("/Users/a12517/Desktop/feedback-prize-2021/train.csv")

train %>% 
  slice_head(n = 30000) -> train

#feature engineering
flesch <- textstat_readability(train$discourse_text, measure = "Flesch.Kincaid")

train <- cbind(train, flesch)

train %>% 
  select(-document) %>% 
  rename(flesch = `Readability Score`) -> train

# feature engineering
lex_div <- textstat_lexdiv(dfm(train$discourse_text))

lex_div %>% 
  tibble %>% 
  bind_cols(train) -> train

train %>% 
  select(-document) -> train

# feature engineering 
gl <- textstat_readability(train$discourse_text, measure = "Coleman.Liau.grade")

train <- cbind(train, gl)

lex_div %>% 
  tibble %>% 
  bind_cols(train) -> train

train %>% 
  select(-document) -> train

#------feature engineering


train %>% 
  select(-(starts_with("document"))) %>% 
  select(-c(1)) %>% 
  rename(TTR = c(1)) %>% 
  select(-c(2,3,4,5,8,9)) -> train


hapax_proportion <- rowSums(dfm(train$discourse_text) == 1) / ntoken(dfm(train$discourse_text))

hapax_proportion %>% 
  as_tibble() %>% 
  rename(hapax = value) %>% 
  bind_cols(train) -> train

#---------------
sevenths = function (x) {
  x = (x*100)
  if (x / 7 == 0) {
     return(as.numeric(floor(x / 7)))
  } else {
    return (as.numeric(floor(x / 7)) * 7) + 7
  }
}
sevenths(train$hapax[1])
map_dbl(train$hapax, sevenths) -> new

train$new <- new

train %>% 
  group_by(discourse_type) %>% 
  rowid_to_column() %>% 
  summarize(ok = mean(new))

train



train %>% 
  group_by(discourse_type) %>% 
  mutate(wanted = median(new)) %>% 
  ungroup() %>% 
  group_by(wanted) %>% 
  tally 
  
train %>% 
  group_by(new) %>% 
  tally



#-----------
train %>% 
  group_by(discourse_type) %>% 
  mutate(wanted = median(hapax*10000),
         wanted = round(wanted, -3)) %>% 
  ungroup()  %>% 
  group_by(wanted) %>% 
  tally

#-----------
library(tidytext)
library(tidylo)

read.csv("common_wods.csv") -> common_words

common_words %>% 
  rename(word = common_words) %>% 
  tibble -> common_words

range(train$hapax)

train %>% 
  select(hapax,
         discourse_text) %>% 
  rowid_to_column() %>% 
  unnest_tokens(word, 
                discourse_text) %>% 
  inner_join(get_stopwords(), by = "word") %>% 
  #distinct(word, .keep_all = TRUE) 
  count(hapax,
        word, 
        sort = TRUE) %>% 
  bind_log_odds(hapax, word, n) %>% 
  group_by(hapax) %>% 
  arrange(-n) %>% 
  select(hapax,
         word,
         log_odds_weighted) %>% 
  ungroup() -> weighted_words

weighted_words %>% 
  pivot_wider(
    names_from = word,
    values_from = log_odds_weighted
  ) -> values

multicomplaints_train %>% 
  left_join(values, )

d#-----------

write.csv(wanted_words, "wanted.words.csv")
wanted_words %>% 
  tibble

names <- c(as.character(1:250))

train %>% 
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


train %>% 
  rowid_to_column() %>% 
  left_join(new_matrix, by = c("rowid")) %>% 
  tibble -> new_raw

new_raw %>% 
  janitor::clean_names() %>% 
  tibble -> new_raw

new_raw %>% 
  tibble

new_raw %>% 
  select(-rowid) %>% 
  write.csv("with_logged.csv")

read.csv("with_logged.csv") -> logged

logged %>% 
  tibble

logged %>% 
  select(-X) -> log

log %>% 
  replace(is.na(.), 0) -> log

#-----------
library(tidymodels)
library(textrecipes)

write.csv(new_raw, "by_tenth.csv")
read.csv("by_tenth.csv") -> tenth

tenth %>% 
  tibble %>% 
  select(-X, -rowid) -> tenth

tenth %>% 
  replace(is.na(.), 0) -> tenth
  
split_tenth <- initial_split(tenth, 
                          strata = discourse_type)

train_tenth <- training(split_tenth)
test_tenth <- testing(split_tenth)

colnames(train_tenth)

train_tenth %>%
  count(discourse_type, sort = TRUE) %>%
  select(n, discourse_type)

colnames(train_tenth)

#-------
m_rec <-
  recipe(discourse_type ~ .,
         data = train_tenth) %>% 
  step_tokenize(discourse_text) %>%
  step_tokenfilter(discourse_text, 
                   max_tokens = 800) %>%
  textrecipes::step_tfidf(discourse_text) %>% 
  themis::step_downsample(discourse_type)

#-------
m_rec

#------
m_folds <- vfold_cv(train_tenth, 
                    10)

m_spec <- multinom_reg(penalty = tune(), 
                       mixture = 1) %>%
  set_mode("classification") %>%
  set_engine("glmnet")

m_spec

#-------
library(hardhat)
sparse_bp <- default_recipe_blueprint(composition = "dgCMatrix")

m_lasso_wf <- workflow() %>%
  add_recipe(m_rec, 
             blueprint = sparse_bp) %>%
  add_model(m_spec)

m_lasso_wf

#--------
s_lambda <- grid_regular(penalty(range = c(-5, 0)), levels = 10)
s_lambda

#--------
m_rs <- tune_grid(
  m_lasso_wf,
  m_folds,
  grid = s_lambda,
  control = control_resamples(save_pred = TRUE)
)
#------

best_acc <- m_rs %>%
  show_best("accuracy")

best_acc

m_rs %>%
  collect_predictions() %>% 
  #f_meas(discourse_type, .pred_class)
  filter(penalty == best_acc$penalty) %>% 
  filter(id == "Fold01") %>% 
  conf_mat(discourse_type, .pred_class) %>% 
  autoplot(type = "heatmap") +
  scale_y_discrete(labels = function(x) str_wrap(x, 20)) +
  scale_x_discrete(labels = function(x) str_wrap(x, 20))

collect_metrics(m_rs)
autoplot(m_rs)

m_rs %>%
  show_best("roc_auc")

best_roc_auc <- select_best(m_rs, "roc_auc")
best_roc_auc

collect_predictions(m_rs, 
                    parameters = best_roc_auc)

collect_predictions(m_rs, 
                    parameters = best_roc_auc) %>% 
  conf_mat(discourse_type, .pred_class) %>% 
  autoplot(type = "heatmap") +
  scale_y_discrete(labels = function(x) str_wrap(x, 20)) +
  scale_x_discrete(labels = function(x) str_wrap(x, 20))

wf_spec_final <- finalize_workflow(m_lasso_wf, 
                                   best_roc_auc)

#----------
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

#---------------
read_it <- function(x) {
  texts = read_lines(x) %>% 
    as.data.frame() %>% 
    na_if("") %>% 
    drop_na() %>% 
    tibble
}

read_it("/Users/a12517/Desktop/feedback-prize-2021/test/0FB0700DAF44.txt") -> first
read_it("/Users/a12517/Desktop/feedback-prize-2021/test/18409261F5C2.txt") -> second
read_it("/Users/a12517/Desktop/feedback-prize-2021/test/D46BCB48440A.txt") -> third
read_it("/Users/a12517/Desktop/feedback-prize-2021/test/D72CB1C11673.txt") -> fourth
read_it("/Users/a12517/Desktop/feedback-prize-2021/test/DF920E0A7337.txt") -> fifth

rbind(first, second, third, fourth, fifth) -> test_text




colnames(test_text) <- "discourse_text"

test_text %>% 
  tibble

#----------- split on punctuation
test_text %>% 
  mutate(discourse_text = str_split(discourse_text, '(?<=[.!?]) +')) %>% 
  unnest(discourse_text) %>% 
  tibble %>% 
  na_if("") %>% 
  drop_na -> text_test

text_test <- test_text
#------------ score test data
library(quanteda)
library(quanteda.textstats)
flesch <- textstat_readability(text_test$discourse_text, 
                               measure = "Flesch.Kincaid")

text_test <- cbind(text_test, flesch)

text_test %>% 
  select(-document) -> text_test

#-----ttr
lex_div <- textstat_lexdiv(dfm(text_test$discourse_text))

lex_div %>% 
  tibble %>% 
  bind_cols(text_test) -> text_test

text_test %>% 
  select(-document) -> text_test

# feature engineering 
gl <- textstat_readability(text_test$discourse_text, 
                           measure = "Coleman.Liau.grade")

text_test <- cbind(text_test, gl)

text_test %>% 
  select(-document) -> text_test

#hapax

hapax_proportion <- rowSums(dfm(text_test$discourse_text) == 1) / ntoken(dfm(text_test$discourse_text))

hapax_proportion %>% 
  as_tibble() %>% 
  rename(hapax = value) %>% 
  bind_cols(text_test) -> text_test


text_test %>% 
  group_by(hapax) %>% 
  tally(sort = T)
# create buckets

train %>% 
  group_by(wanted) %>% 
  tally

text_test %>% 
  mutate(wanted = hapax*10000,
         wanted = round(wanted, -3)) %>% 
  group_by(wanted) %>% 
  tally
