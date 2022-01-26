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
max_words <- 18000
max_length <- 30

kick_rec <- recipe(~ discourse_text, data = cnn_train) %>%
  step_tokenize(discourse_text) %>%
  step_tokenfilter(discourse_text, max_tokens = max_words) %>%
  step_sequence_onehot(discourse_text, sequence_length = max_length) 

kick_prep <- prep(kick_rec)
kick_matrix <- bake(kick_prep, new_data = NULL, composition = "matrix")

dim(kick_matrix)

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

final_history <- fin_model %>%
  fit(
    kick_matrix,
    y_train,
    epochs = 10,
    validation_split = 0.1,
    batch_size = 512
  )

final_history

#---------------prep test data and lables

kick_matrix_test <- bake(kick_prep, new_data = cnn_test, # from other file
                         composition = "matrix")

cnn_test %>% 
  mutate(discourse_type = as.factor(discourse_type),
         labels = as.numeric(discourse_type)) -> test_labels

y_test= keras::to_categorical(test_labels$labels-1)
dim(y_test)

# evaluate final model

fin_model %>% evaluate(kick_matrix_test, y_true)

fin_model %>% predict(matrix_test) %>% k_argmax() -> arrays # not sure about this

#-------- second model, dense model
n_units = 100
model %>% 
  layer_dense(units = n_units, 
              activation = 'relu', 
              input_shape = dim(kick_matrix)[2]) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_dense(units = n_units, activation = 'relu') %>%
  layer_dropout(rate = 0.25) %>%
  layer_dense(units = n_units, activation = 'relu') %>%
  layer_dropout(rate = 0.25) %>%
  layer_dense(units = 7, activation = 'softmax')

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = metric_auc(multi_label = TRUE)
)

model %>% fit(
  kick_matrix,
  y_train,
  epochs = 5, batch_size = 32, verbose = 1, 
  validation_split = 0.1
)



#--------mdoel with binary outcome -- performs well... cnn

train %>% 
  mutate(binary = if_else(labels > 1, 0, 1)) -> train

train %>% 
  tibble %>% 
  head %>% View

max_words <- 2e4
max_length <- 30

kick_rec <- recipe(~ discourse_text, data = train) %>%
  step_tokenize(discourse_text) %>%
  step_tokenfilter(discourse_text, max_tokens = max_words) %>%
  step_sequence_onehot(discourse_text, sequence_length = max_length)

kick_prep <- prep(kick_rec)
kick_matrix <- bake(kick_prep, new_data = NULL, composition = "matrix")

dim(kick_matrix)

final_mod <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words + 1, 
                  output_dim = 16,
                  input_length = max_length) %>%
  layer_conv_1d(filter = 32, kernel_size = 7,
                strides = 1, activation = "relu") %>%
  layer_global_max_pooling_1d() %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

final_mod %>%
  compile(
    optimizer = "adam",
    loss = "binary_crossentropy",
    metrics = c("accuracy")
  )

final_history <- final_mod %>%
  fit(
    kick_matrix,
    train$binary,
    epochs = 10,
    validation_split = 0.1,
    batch_size = 512,
    verbose = FALSE
  )

final_history

kick_matrix_test <- bake(kick_prep, new_data = train,
                         composition = "matrix")

final_res <- keras_predict(final_mod, 
                           kick_matrix_test) %>% k_argmax()

final_res_prob <- keras_predict(final_mod, 
                                kick_matrix_test)

final_res_prob %>% 
  as.array() %>% 
  as_tibble() %>% 
  mutate(event = if_else(V1 >= .5, 1, 0)) %>% 
  bind_cols(train$binary) %>% 
  select(-1) %>% 
  rename(pred = c(1), obs = c(2)) %>% 
  mutate(pred = as.factor(pred),
         obs = as.factor(obs)) %>% 
  f_meas(obs, pred)
  
final_res %>% metrics(binary, 
                      .pred_class, 
                      .pred_1)


