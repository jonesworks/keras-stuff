library(keras)
library(textrecipes)
library(tidyverse)
library(data.table)

fread("/Users/a12517/Desktop/feedback-prize-2021/train.csv") -> train
glimpse(train)

train %>% 
  slice_head(n = 30000) -> train

library(tidymodels)

cnn_split <- initial_split(train, 
                           strata = discourse_type)

cnn_train <- training(cnn_split)
cnn_test <- testing(cnn_split)

cnn_train %>%
  count(discourse_type, sort = TRUE) %>%
  select(n, discourse_type)

#first model -- recipe prep
max_words <- 2e4
max_length <- 30

kick_rec <- recipe(~ discourse_text, 
                   data = cnn_train) %>%
  step_tokenize(discourse_text) %>%
  step_tokenfilter(discourse_text, 
                   max_tokens = 100) %>% #these should be reversed
  step_sequence_onehot(discourse_text, 
                       sequence_length = 300,
                       prefix = "",
                       padding = "post", 
                       truncating = "post")

kick_prep <- prep(kick_rec)
kick_matrix <- bake(kick_prep, 
                    new_data = NULL, 
                    composition = "matrix")

dim(kick_matrix)
kick_matrix %>% 
  head

# prep labels

cnn_train %>% 
  mutate(discourse_type = as.factor(discourse_type),
         labels = as.numeric(discourse_type)) -> train_labels

y_train = keras::to_categorical(train_labels$labels-1)
dim(y_train)


#-----------------------------------------cnn model
fin_model <- keras_model_sequential() %>%
  layer_embedding(input_dim = dim(kick_matrix)[1] + 1, 
                  output_dim = 16,
                  input_length = dim(kick_matrix)[2]) %>%
  layer_conv_1d(filter = 32, kernel_size = 7,
                strides = 1, activation = "relu") %>%
  layer_global_max_pooling_1d() %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 7, activation = "softmax")

fin_model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = "adam",
  #metrics = c('AUC')
  metrics = metric_auc(multi_label = TRUE)
)

final_history <- fin_model %>% # this take forever
  fit(
    kick_matrix,
    y_train,
    epochs = 10,
    validation_split = 0.1,
    batch_size = 512
  )

final_history

#-----------evaluate
cnn_test %>% 
  mutate(discourse_type = as.factor(discourse_type),
         labels = as.numeric(discourse_type)) -> test_labels

cnn_test_labels= keras::to_categorical(test_labels$labels-1)

kick_matrix_test <- bake(kick_prep, 
                         new_data = cnn_test, # from other file
                         composition = "matrix")

fin_model %>% evaluate(kick_matrix_test, cnn_test_labels) # this gave .6 accuracy


#------------------------- keras preprocessing
num_words <- 20000
max_length <- 200
text_vectorization <- layer_text_vectorization(
  max_tokens = num_words, 
  output_sequence_length = max_length, 
)

text_vectorization %>% 
  adapt(cnn_train$discourse_text)

text_vectorization(matrix(cnn_train$discourse_text[1], ncol = 1))

input <- layer_input(shape = c(1), dtype = "string")

output <- input %>% 
  text_vectorization() %>% 
  layer_embedding(input_dim = num_words + 1, 
                  output_dim = 16) %>%
  layer_global_average_pooling_1d() %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dropout(0.2) %>%
  layer_dense(units = 256, activation = "relu") %>%
  layer_dropout(0.2) %>% 
  layer_dense(units = 128, activation = "relu") %>%
  layer_dropout(0.3) %>% 
  layer_dense(units = 7, activation = "softmax")

model <- keras_model(input, output)

model %>% compile(
  optimizer = 'adam',
  loss = 'categorical_crossentropy',
  metrics = metric_auc(multi_label = TRUE)
)

cnn_train %>% 
  mutate(discourse_type = as.factor(discourse_type),
         labels = as.numeric(discourse_type)) -> train_labels

cnn_train_labels= keras::to_categorical(train_labels$labels-1)

history <- model %>% fit(
  cnn_train$discourse_text,
  cnn_train_labels,
  epochs = 10,
  batch_size = 512,
  validation_split = 0.2,
  verbose=2
)

history
# evaluate

cnn_test %>% 
  mutate(discourse_type = as.factor(discourse_type),
         labels = as.numeric(discourse_type)) -> test_labels

cnn_test_labels= keras::to_categorical(test_labels$labels-1)


model %>% evaluate(cnn_test$discourse_text, cnn_test_labels) # this gave .6 accuracy
