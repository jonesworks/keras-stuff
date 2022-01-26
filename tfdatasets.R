library(tidymodels)
library(textrecipes)
library(tfdatasets)

read.csv("/Users/a12517/python_1/all readability.csv") -> all
read.csv("/Users/a12517/python_1/wanted.words.csv") -> words

words %>% 
  pluck("word") -> word_list

glimpse(all)

all %>% 
  select(-X) -> all

all %>% 
  select(-labels, -group_no, -linenumber) -> all

all %>% 
  replace(is.na(.), 0) -> all

cnn_three_split <- initial_split(all, 
                                 strata = discourse_type) 

cnn_three_train <- training(cnn_three_split)
cnn_three_test <- testing(cnn_three_split)

colnames(cnn_three_train)


#----- tokenize one hot 
max_words = 18903
max_length = 600

  
recipe( ~ discourse_text,
          data = cnn_three_train) %>%
  step_tokenize(discourse_text) %>%
  step_ngram(discourse_text, num_tokens = 3) %>%
  step_tokenfilter(discourse_text, 
                   max_tokens = 500) %>%
  step_sequence_onehot(discourse_text, 
                       sequence_length = max_length) %>% 
  prep() %>% 
  bake(new_data = NULL) -> data
  
cnn_three_train %>% 
  cbind(data) -> new_data

new_data %>% 
  glimpse

library(keras)

ft_spec <- new_data %>% 
  feature_spec(discourse_type ~ .) %>% 
  step_numeric_column(flesch, colman_grade, TTR, mean, grade_no,
                      hapax, smog_score, meanWordSyllables, meanSentenceLength, 
                      Wheeler.Smith, Spache, RIX, nWS, count_dale, spach_count) %>% 
  step_numeric_column(starts_with("seq1")) %>% 
  fit()

ft_spec$dense_features()

input <- layer_input_from_dataset(new_data %>% 
    select(-discourse_type))

#output <- input %>% 
#  layer_dense_features(dense_features(ft_spec)) %>% 
#  layer_dense(units = 1, activation = "sigmoid")

output <- input %>%
  layer_dense_features(ft_spec$dense_features()) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 1, "sigmoid")

model <- keras_model(input, output)
model

model %>% 
  compile(
    loss = "categorical_crossentropy", 
    optimizer = "adam",
    metrics = "accuracy"
  )

new_data %>%  
  select(discourse_type) %>% 
  mutate(discourse_type = as.factor(discourse_type),
         discourse_type = as.numeric(discourse_type)) -> train_labels

train_labels %>% 
  mutate(discourse_type = if_else(discourse_type == 3, 1, 0)) -> train_labels

train_labels %>% 
  group_by(discourse_type) %>% 
  tally


history <- model %>% 
  fit(
    x = new_data %>% 
      select(-discourse_type),
    y = train_labels$discourse_type, 
    epochs = 5, 
    validation_split = 0.2
  )

history

#--------evaluate
recipe( ~ discourse_text,
        data = cnn_three_test) %>%
  step_tokenize(discourse_text) %>%
  step_ngram(discourse_text, num_tokens = 3) %>%
  step_tokenfilter(discourse_text, 
                   max_tokens = 500) %>%
  step_sequence_onehot(discourse_text, 
                       sequence_length = max_length) %>% 
  prep() %>% 
  bake(new_data = NULL) -> test_data

cnn_three_test %>% 
  cbind(test_data) -> new_data

new_data %>% 
  dim

new_data %>% 
  select(discourse_type) %>% 
  mutate(discourse_type = as.factor(discourse_type),
         discourse_type = as.numeric(discourse_type)) -> cnn_test_labels

cnn_test_labels %>% 
  mutate(discourse_type = if_else(discourse_type == 3, 1, 0)) -> test_labels

test_labels %>% 
  group_by(discourse_type) %>% 
  tally

model %>% evaluate(new_data %>% 
                   select(-discourse_type), 
                   test_labels$discourse_type) # 50 auc

model %>% 
  predict(new_data %>% 
            select(-discourse_type)) %>% 
  bind_cols(test_labels$binary) -> preds

preds %>% 
  tibble %>% 
  as.data.frame.table() %>% 
  select(-Var1, -Var2) -> arrays

arrays %>% 
  head

colnames(arrays) <- "preds"

arrays %>% 
  tibble %>% 
  mutate(preds = if_else(preds > .5, 1, 0)) %>% 
  group_by(preds) -> arrays

arrays %>% 
  group_by(preds) %>% 
  tally

test_labels %>% 
  group_by(discourse_type) %>% 
  tally

arrays %>% 
  bind_cols(test_labels) %>% 
  tibble %>% 
  group_by(discourse_type,  preds) %>% 
  tally %>% View

arrays %>% 
  bind_cols(test_labels) %>% 
  tibble %>% 
  mutate(across(everything(), factor))  -> new_arrays

new_arrays %>% 
  conf_mat(discourse_type, preds) %>% 
  autoplot(type = "heatmap")

new_arrays %>% 
  f_meas(discourse_type, preds)
