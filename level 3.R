library(tidymodels)
library(textrecipes)

cnn_three_split <- initial_split(train, 
                                 strata = discourse_type) 

cnn_three_train <- training(cnn_three_split)
cnn_three_test <- testing(cnn_three_split)

max_words = 2e4
max_length = 600

kick_rec <- recipe(~ discourse_text, data = cnn_three_train) %>% 
  step_tokenize(discourse_text) %>%
  step_tokenfilter(discourse_text, max_tokens = max_words) %>%
  step_sequence_onehot(discourse_text, sequence_length = max_length)

kick_prep <- prep(kick_rec)
kick_matrix <- bake(kick_prep, new_data = NULL, composition = "matrix")

library(keras)

dim(kick_matrix)

model <- keras_model_sequential() 

model %>% 
  layer_embedding(input_dim = max_words + 1, 
                  output_dim = 16,
                  input_length = max_length) %>%
  layer_conv_1d(filter = 32, kernel_size = 7,
                strides = 1, activation = "relu") %>%
  layer_global_max_pooling_1d() %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model

model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

cnn_three_train %>% 
  select(discourse_type) %>% 
  mutate(discourse_type = as.factor(discourse_type),
         labels = as.numeric(discourse_type)) -> train_labels

train_labels %>% 
  mutate(binary = if_else(labels == 3, 1, 0)) -> train_labels

train_labels %>% 
  group_by(labels, binary) %>% 
  tally
  
  
history <- model %>% 
  fit(
    kick_matrix, 
    test_labels$binary, 
    epochs = 10,
    validation_split = 0.1,
    batch_size = 512
)

history

#--------evaluate
cnn_three_test %>% 
  select(discourse_type) %>% 
  mutate(discourse_type = as.factor(discourse_type),
         labels = as.numeric(discourse_type)) -> cnn_test_labels

cnn_test_labels %>% 
  mutate(binary = if_else(labels == 3, 1, 0)) -> test_labels

test_labels %>% 
  group_by(labels, binary) %>% 
  tally

matrix_test <- bake(kick_prep, 
                           new_data = cnn_three_test, 
                           composition = "matrix")

model %>% evaluate(matrix_test, test_labels$binary) # 50 auc

model %>% predict(matrix_test) %>% bind_cols(test_labels$binary) %>% View  # not sure about this

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
new_arrays$preds <- factor(new_arrays$preds, levels = c(levels(new_arrays$preds), "2"))

new_arrays$preds <- factor(new_arrays$preds, levels = c("1", "2", "3", "4", "5", "6", "7"))

new_arrays %>% 
  conf_mat(labels, preds) %>% 
  autoplot(type = "heatmap")
