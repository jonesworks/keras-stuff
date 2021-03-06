#---------
# First, let's define a vision model using a Sequential model.
# This model will encode an image into a vector.
vision_model <- keras_model_sequential() 
vision_model %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu', padding = 'same',
                input_shape = c(224, 224, 3)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = 'relu', padding = 'same') %>% 
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 256, kernel_size = c(3, 3), activation = 'relu', padding = 'same') %>% 
  layer_conv_2d(filters = 256, kernel_size = c(3, 3), activation = 'relu') %>% 
  layer_conv_2d(filters = 256, kernel_size = c(3, 3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_flatten()

# Now let's get a tensor with the output of our vision model:
image_input <- layer_input(shape = c(224, 224, 3))
encoded_image <- image_input %>% vision_model

# Next, let's define a language model to encode the question into a vector.
# Each question will be at most 100 word long,
# and we will index words as integers from 1 to 9999.
question_input <- layer_input(shape = c(100), dtype = 'int32')
encoded_question <- question_input %>% 
  layer_embedding(input_dim = 10000, output_dim = 256, input_length = 100) %>% 
  layer_lstm(units = 256)

# Let's concatenate the question vector and the image vector then
# train a logistic regression over 1000 words on top
output <- layer_concatenate(c(encoded_question, encoded_image)) %>% 
  layer_dense(units = 1000, activation='softmax')

# This is our final model:
vqa_model <- keras_model(inputs = c(image_input, question_input), outputs = output)