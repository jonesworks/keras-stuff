# branching from conv network
library(tidymodels)
library(textrecipes)

cnn_three_split <- initial_split(train, 
                                 strata = discourse_type) 

cnn_three_train <- training(cnn_three_split)
cnn_three_test <- testing(cnn_three_split)

cnn_three_train %>%
  count(discourse_type, sort = TRUE) %>%
  select(n, discourse_type)

max_words <- 12000
max_length <- 30

cnn_three_rec <-
  recipe( discourse_type ~ 
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
          data = cnn_three_train) %>%  
  step_tokenize(discourse_text) %>%
  step_tokenfilter(discourse_text, 
                   max_tokens = max_words) %>%
  step_lda(discourse_text) #%>% 
  #themis::step_downsample(discourse_type) %>% 
  #step_rm(discourse_type)

prepped_three <- prep(cnn_three_rec)

prepped_three_matrix <- bake(prepped_three, 
                             new_data = NULL, 
                             composition = "matrix")

dim(prepped_three_matrix)


#----- model
library(keras)
cnn_mod <- keras_model_sequential()
cnn_mod %>% 
  layer_conv_1d(filters = 64, 
                kernel_size = 3, 
                strides = 1,
                activation = 'relu',
                input_shape = c(NULL, 19, 1)) %>% 
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
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 36, activation = 'relu') %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 7, activation = 'softmax')

cnn_mod

cnn_mod %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(),
  metrics = metric_auc(multi_label = TRUE)
)

cnn_three_train %>% 
  mutate(discourse_type = as.factor(discourse_type),
         labels = as.numeric(discourse_type)) -> cnn_three_train_label

y_train = keras::to_categorical(cnn_three_train_label$labels-1)
y_train %>% 
  head()

third_his <- cnn_mod %>%
  fit(
    prepped_three_matrix,
    y_train,
    epochs = 10,
    validation_split = 0.1,
    batch_size = 512,
  )

third_his

#----------

cnn_three_test %>% 
  select(discourse_type) %>% 
  mutate(discourse_type = as.factor(discourse_type),
         labels = as.numeric(discourse_type)) -> cnn_test_labels


y_true= keras::to_categorical(cnn_test_labels$labels-1)

prepped_three_test <- bake(prepped_three, 
                             new_data = cnn_three_test, 
                             composition = "matrix")

cnn_mod %>% evaluate(prepped_three_test, y_true) # 50 auc

cnn_mod %>% predict(prepped_three_test) %>% k_argmax() -> arrays # not sure about this

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

#------------------by grade --- this didn't really improve things...
cnn_three_train %>% 
  group_by(grade_no) %>% 
  tally

cnn_three_train %>% 
  filter(grade_no == 6) -> sixth_train

cnn_three_test %>% 
  filter(grade_no == 6) -> sixth_test

#-----------
sixth_rec <-
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
          data = sixth_train) %>%  
  step_tokenize(discourse_text) %>%
  step_tokenfilter(discourse_text, 
                   max_tokens = max_words) %>%
  step_lda(discourse_text) 

prepped_sixth <- prep(sixth_rec)

prepped_sixth_matrix <- bake(prepped_sixth, 
                             new_data = NULL, 
                             composition = "matrix")

dim(prepped_sixth_matrix)


#----- model

cnn_mod_sixth <- keras_model_sequential()
cnn_mod_sixth %>% 
  layer_conv_1d(filters = 64, 
                kernel_size = 3, 
                strides = 1,
                activation = 'relu',
                input_shape = c(19, 1)) %>% 
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
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 36, activation = 'relu') %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 7, activation = 'softmax')

cnn_mod_sixth

cnn_mod_sixth %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(),
  metrics = metric_auc(multi_label = TRUE)
)

sixth_train %>% 
  mutate(discourse_type = as.factor(discourse_type),
         labels = as.numeric(discourse_type)) -> sixth_train_label

y_train = keras::to_categorical(sixth_train_label$labels-1)
y_train %>% 
  head()

sixth_his <- cnn_mod_sixth %>%
  fit(
    prepped_sixth_matrix,
    y_train,
    epochs = 40,
    validation_split = 0.1,
    batch_size = 512,
  )

sixth_his

#----------

sixth_test %>% 
  select(discourse_type) %>% 
  mutate(discourse_type = as.factor(discourse_type),
         labels = as.numeric(discourse_type)) -> cnn_sixth_labels


y_true= keras::to_categorical(cnn_sixth_labels$labels-1)

prepped_sixth_test <- bake(prepped_sixth, 
                           new_data = sixth_test, 
                           composition = "matrix")

cnn_mod_sixth %>% evaluate(prepped_sixth_test, y_true) # 50 auc

cnn_mod_sixth %>% predict(prepped_sixth_test) %>% k_argmax() -> arrays # not sure about this

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

sixth_test %>% 
  group_by(discourse_type) %>% 
  tally

arrays %>% 
  bind_cols(cnn_sixth_labels) %>% 
  tibble %>% 
  group_by(discourse_type, labels, preds) %>% 
  tally %>% View
