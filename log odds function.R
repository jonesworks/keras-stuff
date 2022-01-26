# more feature engineering 
mws <- textstat_readability(train$discourse_text, 
                            measure = "meanWordSyllables")

train <- cbind(train, mws)

train %>% 
  tibble() %>% 
  select(-document) -> train


# more feature engineering 
msl <- textstat_readability(train$discourse_text, 
                            measure = "meanSentenceLength")

train <- cbind(train, msl)

train %>% 
  tibble() %>% 
  select(-document) -> train


# more feature engineering 
ws <- textstat_readability(train$discourse_text, 
                            measure = "Wheeler.Smith")

train <- cbind(train, ws)

train %>% 
  tibble() %>% 
  select(-document) -> train

# more feature engineering 
sp <- textstat_readability(train$discourse_text, 
                           measure = "Spache")

train <- cbind(train, sp)

train %>% 
  tibble() %>% 
  select(-document) -> train


# more feature engineering 
rix <- textstat_readability(train$discourse_text, 
                           measure = "RIX")

train <- cbind(train, rix)

train %>% 
  tibble() %>% 
  select(-document) -> train


# more feature engineering 
nws <- textstat_readability(train$discourse_text, 
                            measure = "nWS")

train <- cbind(train, nws)

colnames(train)

train %>% 
  tibble() %>% 
  select(-document) -> train


#-------------z
library(tidymodels)
ssplit <- initial_split(train, 
                        strata = discourse_type) 

ttrain <- training(ssplit)
ttest <- testing(ssplit)

ttrain %>% 
  mutate(c_grade = floor(colman_grade)) -> ttrain

ttest %>% 
  mutate(c_grade = floor(colman_grade)) -> ttest

summary(ttrain)

ttrain %>%
  count(discourse_type, sort = TRUE) %>%
  select(n, discourse_type)


ttrain %>% 
  group_by(flesch) %>% 
  tally

#---------- not sure this help more than to overfit
library(tidylo)
max_log_odds <- function (df) {
  df %>% 
    tibble %>% 
    select(flesch,
           discourse_type, 
           discourse_text) %>% 
    mutate(flesch_ceiling = ceiling(flesch)) %>% 
    rowid_to_column() %>% 
    unnest_tokens(words, 
                  discourse_text) %>% 
    count(discourse_type,
          flesch_ceiling,
          words, 
          sort = TRUE) %>% 
    bind_log_odds(flesch_ceiling, 
                  words, n) %>% 
    group_by(flesch_ceiling, 
             discourse_type) %>% 
    arrange(-log_odds_weighted) %>% 
    slice_head(n = 20) %>% 
    ungroup() %>% 
    select(-n) %>% 
    pivot_wider(
      names_from = words,
      values_from = log_odds_weighted
    ) %>% 
    replace(is.na(.),0)
}

max_log_odds(ttrain) -> log_df

log_df %>% 
  tibble %>% 
  head(20)

ttrain %>% 
  mutate(flesch_ceiling = ceiling(flesch)) %>% 
  left_join(log_df, 
            by = c("flesch_ceiling", 
                 "discourse_type")) %>% 
  tibble -> ttrain

#-----------

add <- function (df, var) {
  df %>% 
    rename({{ var }} := mean_logs) %>% 
    tibble
}
add(new_data, c_log)

#------------ this is the best function but not used

l_odds <- function (df, var, var1) {
  df %>% 
    as_tibble() %>% 
    select({{ var }},
           discourse_type, 
           discourse_text) %>% 
    rowid_to_column() %>% 
    unnest_tokens(words, 
                  discourse_text) %>% 
    count({{ var }},
          discourse_type,
          words, 
          sort = TRUE) %>% 
    bind_log_odds({{ var }}, words, n) %>% 
    group_by({{ var }}, discourse_type) %>% 
    mutate(mean_logs= min(log_odds_weighted)) %>% 
    select(-n, -log_odds_weighted, -words) %>% 
    unique() %>% 
    rename({{ var1 }} := mean_logs) %>% 
    tibble 
}

l_odds(ttrain, 
       c_grade, 
       c_logs) -> new_data

ttrain %>% 
  left_join(new_data, 
            by = c("c_grade", 
                   "discourse_type")) -> ttrain

l_odds(ttrain, 
       mean, 
       f_logs) -> new_data

ttrain %>% 
  left_join(new_data, 
            by = c("mean", 
                   "discourse_type")) -> ttrain

colnames(ttrain)

ttrain[sapply(ttrain, is.infinite)] <- 0

sum(is.infinite(ttrain$c_logs)) 


#----- endo of logs odds





colnames(ttrain)
#---------

tthree_rec <-
  recipe( ~ 
            hapax +
            TTR +
            flesch +
            mean +
            grade_no + 
            fleisch_type +
            colman_grade +
            scrabble_score +
            smog_score +
            entropy +
            c_grade +
            nWS +
            Spache +
            Wheeler.Smith +
            meanSentenceLength +
            meanWordSyllables +
            RIX +
            discourse_text,
          data = ttrain) %>% 
  step_tokenize(discourse_text) %>%
  step_tokenfilter(discourse_text, 
                   max_tokens = 10000) %>% 
  step_lda(discourse_text) 

prepped_tthree_rec <- prep(tthree_rec)

prepped_tthree_rec_matrix <- bake(prepped_tthree_rec, 
                             new_data = NULL, 
                             composition = "matrix")

dim(prepped_tthree_rec_matrix)


#----- model

cnn_mod <- keras_model_sequential()
cnn_mod %>% 
  layer_conv_1d(filters = 64, 
                kernel_size = 3, 
                strides = 1,
                activation = 'relu',
                input_shape = c(27, 1)) %>% 
  layer_conv_1d(filters = 64, 
                kernel_size = 3, 
                activation = 'relu') %>% 
  layer_max_pooling_1d(pool_size = 3) %>% 
  layer_conv_1d(filters = 128, 
                kernel_size = 3, 
                activation = 'relu') %>% 
  layer_conv_1d(filters = 128, 
                kernel_size = 3, 
                activation = 'relu') %>% 
  layer_global_average_pooling_1d() %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 36, activation = 'relu') %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 7, activation = 'softmax')

cnn_mod

cnn_mod %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(),
  metrics = metric_auc(multi_label = TRUE)
)

ttrain %>% 
  mutate(discourse_type = as.factor(discourse_type),
         labels = as.numeric(discourse_type)) -> ttrain_label

y_train = keras::to_categorical(ttrain_label$labels-1)
y_train %>% 
  head()

third_his <- cnn_mod %>%
  fit(
    prepped_tthree_rec_matrix,
    y_train,
    epochs = 20,
    validation_split = 0.1,
    batch_size = 512,
  )

third_his

#----------


ttest %>% 
  select(discourse_type) %>% 
  mutate(discourse_type = as.factor(discourse_type),
         labels = as.numeric(discourse_type)) -> ttest_labels


y_true= keras::to_categorical(ttest_labels$labels-1)


l_odds(ttest, c_grade, c_logs) -> new_data

ttest %>% 
  left_join(new_data, by = c("c_grade", "discourse_type")) -> ttest

l_odds(ttest, mean, f_logs) -> new_data

ttest %>% 
  left_join(new_data, by = c("mean", "discourse_type")) -> ttest

colnames(ttest)

ttest[sapply(ttest, is.infinite)] <- 0

sum(is.infinite(ttest$c_logs)) 


tthree_test <- bake(prepped_tthree_rec, 
                           new_data = ttest, 
                           composition = "matrix")

cnn_mod %>% evaluate(tthree_test, y_true) # 50 auc

cnn_mod %>% predict(tthree_test) %>% k_argmax() -> arrays # not sure about this

arrays %>% 
  as.array() %>% 
  tibble %>% 
  as.data.frame.table() %>% 
  select(-Var1, -Var2) ->arrays

colnames(arrays) <- "preds"

arrays %>% 
  mutate(preds = preds + 1) -> arrays

arrays %>% 
  mutate(preds = as.factor(preds)) %>% 
  group_by(preds) %>% 
  tally

cnn_three_test %>% 
  group_by(discourse_type) %>% 
  tally

arrays %>% 
  bind_cols(cnn_test_labels) %>% 
  tibble %>% 
  group_by(discourse_type, labels, preds) %>% 
  tally %>% View

arrays %>% 
  bind_cols(cnn_test_labels) %>% 
  tibble %>% 
  mutate(across(everything(), factor))  -> new_arrays

levels(new_arrays$preds)
new_arrays$preds <- factor(new_arrays$preds, levels = c(levels(new_arrays$preds), "6"))

new_arrays$preds <- factor(new_arrays$preds, levels = c("1", "2", "3", "4", "5", "6", "7"))

new_arrays %>% 
  conf_mat(labels, preds) %>% 
  autoplot(type = "heatmap")


