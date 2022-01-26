library(tidytext)
library(quanteda)
library(quanteda.textstats) 
library(data.table)
library(koRpus)
library(tidyverse)
library(textrecipes)
library(keras)

train <- fread("/Users/a12517/Desktop/feedback-prize-2021/train.csv")

train %>% 
  slice_head(n = 30000) -> train

train %>% 
  group_by(id) %>% 
  mutate(group_no = cur_group_id())  %>% 
  select(group_no,
         discourse_text, 
         discourse_type) -> train

#-------text analysis
read_level = function (x) {
  x = textstat_readability(x, measure = "Flesch.Kincaid")
  x = x %>%
    rename(Input = document, "Readability Score" = Flesch.Kincaid)
}

flesch <- read_level(train$discourse_text)

flesch %>% 
  tibble

train <- cbind(train, flesch)

train %>% 
  tibble

train %>% 
  rename(flesch = `Readability Score`) %>% 
  group_by(group_no) %>% 
  mutate(mean = mean(flesch))-> train

train %>% 
  tibble

train %>% 
  mutate(grade_no = floor(mean)) -> train

train %>% 
  group_by(discourse_type) %>% 
  mutate(fleisch_type = mean(flesch)) %>% 
  tibble -> train

train %>% 
  group_by(grade_no, discourse_type) %>% 
  count 

train %>% 
  mutate(
    grade_no = case_when(
      grade_no >= 12 ~ 12,
      grade_no <= 6 ~ 6,
      TRUE ~ grade_no)
  ) %>% 
  group_by(grade_no) -> train

trainlex_div <- textstat_lexdiv(dfm(train$discourse_text))

trainlex_div %>% 
  tibble %>% 
  bind_cols(train) -> train

train %>% 
  select(-Input, -document) -> train

train %>% 
  group_by(group_no, mean) %>% 
  count 

#------more readability scores
grade_level = function (x) {
  x = textstat_readability(x, measure = "Coleman.Liau.grade")
  x = x %>%
    rename(Input = document, "Readability Score" = Flesch.Kincaid)
}

gl <- textstat_readability(train$discourse_text, 
                           measure = "Coleman.Liau.grade")

train <- cbind(train, gl)

train %>% 
  tibble() %>% 
  select(-document) -> train

train %>% 
  rename(colman_grade = Coleman.Liau.grade) -> train

train %>% 
  group_by(grade_no, discourse_type) %>% 
  summarize(ttr = mean(TTR),
            flesch = mean(flesch),
            col = mean(colman_grade)) %>% 
  tibble()

#hapaxproporaiton

# as a proportion
# hapaxes per document

hapax_proportion <- rowSums(dfm(train$discourse_text) == 1) / 
  ntoken(dfm(train$discourse_text))

hapax_proportion %>% 
  as_tibble() %>% 
  rename(hapax = value) %>% 
  bind_cols(train) -> train

#--------------------------
scrabble_level = function (x) {
  x = textstat_readability(x, measure = "Scrabble")
  x = x %>%
    rename(Input = document, scrabble_score = Scrabble)
}


scrabble <- scrabble_level(train$discourse_text)

scrabble %>% 
  tibble

train <- cbind(train, scrabble)

train %>% 
  tibble

train %>% 
  select(-Input) %>% 
  tibble() -> train

#-----smog measure
smog_level = function (x) {
  x = textstat_readability(x, measure = "SMOG")
  x = x %>%
    rename(Input = document, smog_score = SMOG)
}

smog <- smog_level(train$discourse_text)

smog %>% 
  tibble

train <- cbind(train, smog)

train %>% 
  tibble

train %>% 
  select(-Input) %>% 
  tibble() -> train
#--------------------
entropy <- textstat_entropy(dfm(train$discourse_text))

entropy %>% 
  tibble

train <- cbind(train, entropy)

train %>% 
  tibble

train %>% 
  select(-document) %>% 
  tibble() -> train
#-----------------------
library(tidymodels)
library(textrecipes)
cnn_two_split <- initial_split(train, strata = discourse_type)

cnn_two_train <- training(cnn_two_split)
cnn_two_test <- testing(cnn_two_split)

cnn_two_train %>%
  count(discourse_type, sort = TRUE) %>%
  select(n, discourse_type)

library(skimr)
skim(cnn_two_train)

max_words <- 1800
max_length <- 150

cnn_two_rec <-
  recipe( ~ 
            discourse_text +
            hapax +
            mean + 
            fleisch_type + 
            scrabble_score + 
            smog_score + 
            entropy +
            flesch + 
            colman_grade +
            TTR, 
          data = cnn_two_train) %>%  
  step_tokenize(discourse_text) %>%
  step_tokenfilter(discourse_text, 
                   max_tokens = max_words) %>%
  step_sequence_onehot(discourse_text, 
                       sequence_length = max_length) %>% 
  step_shuffle(all_predictors())

prepped <- prep(cnn_two_rec)
prepped_matrix <- bake(prepped, 
                       new_data = NULL, 
                       composition = "matrix")

dim(prepped_matrix)

two_model <- keras_model_sequential() 

two_model %>% 
  layer_dense(units =100, 
              activation = 'relu', 
              input_shape = dim(prepped_matrix)[2]) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 36, activation = 'relu') %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 7, activation = 'softmax')

two_model

two_model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(),
  metrics = metric_auc(multi_label = TRUE)
)

cnn_two_train %>% 
  mutate(discourse_type = as.factor(discourse_type),
         labels = as.numeric(discourse_type)) -> cnn_two_train

y_train = keras::to_categorical(cnn_two_train$labels-1)
y_train %>% 
  head()

dense_model <- two_model %>% 
  fit(
  prepped_matrix, 
  y_train, 
  epochs = 5, 
  batch_size = 32, 
  verbose = 1, 
  validation_split = 0.1
)

dense_model

#---------

cnn_two_test %>% 
  select(discourse_type) %>% 
  mutate(discourse_type = as.factor(discourse_type),
         labels = as.numeric(discourse_type)) -> test_labels

y_true= keras::to_categorical(test_labels$labels-1)

matrix_test <- bake(prepped, 
                    new_data = cnn_two_test,
                    composition = "matrix")

two_model %>% evaluate(matrix_test, y_true)

two_model %>% predict(matrix_test) %>% k_argmax() -> arrays # not sure about this

arrays %>% 
  as.array() %>% 
  tibble %>% 
  as.data.frame.table() %>% 
  select(-Var1, -Var2) ->arrays

colnames(arrays) <- "preds"

arrays %>% 
  mutate(preds = preds + 1) -> arrays

#model %>% predict(x) %>% `>`(0.5) %>% k_cast("int32")

#----vip--- this doens't work

pred_wrapper <- function(object, newdata) {
  predict(object, x = as.matrix(newdata)) %>%
    as.vector()
}

class(prepped_matrix)

set.seed(102)  # for reproducibility
p1 <- vip(
  object = model,                     # fitted model
  method = "permute",                 # request permutation-based VI scores
  num_features = ncol(prepped_matrix),       # default only plots top 10 features
  pred_wrapper = pred_wrapper,            # user-defined prediction function
  target = y_train,            # name of the target variable column
  metric = "rsquared",                # evaluation metric
  train = as.data.frame(prepped_matrix),     # training data
  # progress = "text"                 # request a text-based progress bar
)
print(p1)  # display plot




#---- new rec for model

cnn_two_rec_two <-
  recipe( ~ discourse_text +
            hapax +
            mean + 
            fleisch_type + 
            scrabble_score + 
            smog_score + 
            entropy +
            flesch + 
            colman_grade +
            TTR, 
          data = cnn_two_train) %>% 
  step_tokenize(discourse_text) %>% 
  step_tokenfilter(discourse_text, max_tokens = max_words) %>%
  step_sequence_onehot(discourse_text, sequence_length = max_length) 

prepped_two <- prep(cnn_two_rec_two)
prepped_matrix_two <- bake(prepped_two, 
                           new_data = NULL, 
                           composition = "matrix")

dim(prepped_matrix_two)

#---------
final_mod <- keras_model_sequential()

final_mod %>% 
  layer_conv_1d(filters = 32, 
                kernel_size = 7, 
                activation = 'relu',
                input_shape = c(159, 1))  %>% #number o columns...
  layer_global_max_pooling_1d() %>%
  layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 64, activation = "relu") %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 7, activation = "softmax")

final_mod %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(),
  metrics = metric_auc(multi_label = TRUE)
)

y_train_two = keras::to_categorical(cnn_two_train$labels-1)
dim(y_train_two)


final_history <- final_mod %>%
  fit(
    prepped_matrix_two,
    y_train_two,
    epochs = 5,
    validation_split = 0.1,
    batch_size = 36
  )

final_history

#-------------------------------------------------

cnn_two_test %>% 
  select(discourse_type) %>% 
  mutate(discourse_type = as.factor(discourse_type),
         labels = as.numeric(discourse_type)) -> test_labels


y_true= keras::to_categorical(test_labels$labels-1)

final_mod %>% evaluate(matrix_test, y_true) # 50 auc

final_mod %>% predict(matrix_test) %>% k_argmax() -> arrays # not sure about this

arrays %>% 
  as.array() %>% 
  tibble %>% 
  as.data.frame.table() %>% 
  select(-Var1, -Var2) ->arrays

colnames(arrays) <- "preds"

arrays %>% 
  mutate(preds = preds + 1) -> arrays


#------------------------------
#predictions===============================
matrix_test <- bake(prepped, 
                    new_data = cnn_two_test,
                    composition = "matrix")

classes_first <- model %>% 
  predict(matrix_test)

classes_first %>% 
  tibble()

classes_first %>% 
  tibble() %>% 
  as.data.frame.table %>% 
  select(-1,-2) -> classes_second

colnames(classes_second) <- c('1', "2", "3", "4", "5", "6", "7")

classes_second %>% 
  tibble

new_classes = colnames(classes_second)[max.col(classes_second, ties.method = "first")]  

new_classes %>% 
  tibble

cnn_two_test %>% 
  select(discourse_type) %>% 
  mutate(discourse_type = as.factor(discourse_type),
         labels = as.numeric(discourse_type)) -> test_labels


y_true= keras::to_categorical(test_labels$labels-1)
metric_categorical_crossentropy(y_true, classes_first) %>% as.array()


