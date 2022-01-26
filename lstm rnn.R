library(tidymodels)
library(textrecipes)
library(data.table)

train <- fread("/Users/a12517/Desktop/feedback-prize-2021/train.csv")

train %>% 
  tibble

tnn_three_split <- initial_split(train, 
                                 strata = discourse_type) 

tnn_three_train <- training(tnn_three_split)
tnn_three_test <- testing(tnn_three_split)

max_words = 18903
max_length = 600


recipe( ~ discourse_text,
        data = tnn_three_train) %>%
  step_tokenize(discourse_text) %>%
  step_ngram(discourse_text, num_tokens = 3) %>%
  step_tokenfilter(discourse_text, 
                   max_tokens = max_words) %>%
  step_sequence_onehot(discourse_text, 
                       sequence_length = max_length) -> recipe_new
  
recipe_new %>% 
  prep() %>% 
  bake(new_data = NULL,
       composition = "matrix") -> data


library(keras)

dim(data)

model <- keras_model_sequential() 

model %>% 
  layer_embedding(input_dim = max_words + 1, 
                  output_dim = 64,
                  input_length = max_length) %>%
  bidirectional(layer_lstm(units = 64)) %>%
  layer_dense(units = 64, activation = 'relu') %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 7, activation = 'softmax')


model

model %>% 
  compile(
    loss = "categorical_crossentropy", 
    optimizer = "adam",
    metrics = "accuracy"
)

tnn_three_train %>% 
    select(discourse_type) %>% 
    mutate(discourse_type = as.factor(discourse_type),
           labels = as.numeric(discourse_type)) -> train_labels
  
y_train = keras::to_categorical(train_labels$labels-1)
head(y_train)
  
history <- model %>% 
    fit(
      data, 
      y_train, 
      epochs = 10,
      validation_split = 0.1,
      batch_size = 512
    )
  
history

