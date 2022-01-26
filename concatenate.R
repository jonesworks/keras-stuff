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
  bake(new_data = NULL, composition = "matrix") -> data

data %>% 
  glimpse

library(keras)

text_one_hot <- layer_input(max_length)
text_as_int <- layer_input(max_words)

vec_1 <- text_one_hot %>%
  layer_dense(100)

vec_2 <- layer_embedding(
  input_dim = max_words, 
  output_dim = 128, 
  input_length = max_length
) %>%
  layer_lstm(128)

out <- layer_concatenate(list(vect_1, vec_2))

model <- keras_model(list(input_1, input_2))






max_words <- 20
nb_words <- 1000

text_one_hot <- layer_input(nb_words)
text_as_int <- layer_input(max_words)

vec_1 <- text_one_hot %>%
  layer_dense(100)

vec_2 <- layer_embedding(
  input_dim = nb_words, output_dim = 128, 
  input_length = max_words
) %>%
  layer_lstm(128)

out <- layer_concatenate(list(vect_1, vec_2))





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
