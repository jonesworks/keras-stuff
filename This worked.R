library(tidymodels)
library(textrecipes)
library(tfdatasets)
library(keras)

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

max_words = 500
max_length = 100


rec <- recipe( ~ discourse_text,
               data = cnn_three_train) %>%
  step_tokenize(discourse_text) %>%
  step_ngram(discourse_text, num_tokens = 3) %>%
  step_tokenfilter(discourse_text, 
                   max_tokens = max_words) %>%
  step_sequence_onehot(discourse_text, 
                       sequence_length = max_length)

prep(rec) %>% 
  bake(new_data = NULL, composition = "matrix") -> data


#----recipe two

rec_two <- recipe( ~ 
                     flesch + colman_grade + TTR + mean + grade_no +
                     hapax + smog_score + meanWordSyllables + meanSentenceLength + 
                     Wheeler.Smith + Spache + RIX + nWS + count_dale + spach_count,
                   data = cnn_three_train) %>%
  step_normalize(all_numeric_predictors())

prep(rec_two) %>% 
  bake(new_data = NULL, composition = "matrix") -> data_two



# This model will encode an image into a vector.
dense_model <- keras_model_sequential() 

model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 16, 
              activation = 'relu', 
              dim(data_two)[2]) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 16, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_flatten()

# Now let's get a tensor with the output of our vision model:
data_input <- layer_input(shape = dim(data_two)[2])
encoded_image <- data_input %>% model

# Next, let's define a language model to encode the question into a vector.
# Each question will be at most 100 word long,
# and we will index words as integers from 1 to 9999.

question_input <- layer_input(shape = max_length, 
                              dtype = 'int32')

encoded_question <- question_input %>% 
  layer_embedding(input_dim = max_words +1, 
                  output_dim = 16, 
                  input_length = max_length) %>% 
  layer_lstm(units = 16)

# Let's concatenate the question vector and the image vector then
# train a logistic regression over 1000 words on top
output <- layer_concatenate(c(encoded_question, 
                              encoded_image)) %>% 
  layer_dense(units = 7, activation='softmax')

# This is our final model:
vqa_model <- keras_model(inputs = c(data_input, 
                                    question_input), 
                         outputs = output)

vqa_model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = "accuracy"
)


cnn_three_train %>% 
  mutate(discourse_type = as.factor(discourse_type),
         labels = as.numeric(discourse_type)) -> train_labels

y_train= keras::to_categorical(train_labels$labels-1)
dim(y_train)


history <- vqa_model %>% 
  fit(
    list(data_two, 
         data),
    y_train, 
    batch_size = 64, 
    epochs = 5,
    validation_split = 0.1
  )

history
