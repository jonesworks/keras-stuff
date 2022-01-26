max_words <- 2e4
max_length <- 30

final_mod <- keras_model_sequential() 

final_mod() %>% 
  layer_embedding(input_dim = max_words + 1, 
                  output_dim = 16,
                  input_length = max_length) %>%
  layer_conv_1d(filter = 32, kernel_size = 7,
                strides = 1, activation = "relu") %>%
  layer_global_max_pooling_1d() %>%
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dropout(rate = 0.25) 

n_units = 100
model_four <- keras_model_sequential() 

model_five <- model_four %>% 
  layer_dense(units = n_units, 
              activation = 'relu', 
              input_shape = max_length) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_dense(units = n_units, activation = 'relu') %>%
  layer_dropout(rate = 0.25) %>%
  layer_dense(units = n_units, activation = 'relu') %>%
  layer_dropout(rate = 0.25) 

shared_lstm <- layer_dense(units = 64)

encoded_a <- model_five %>% shared_lstm
encoded_b <- final_mod %>% shared_lstm

predictions <- layer_concatenate(c(encoded_a, encoded_b), axis=-1)      
                          
