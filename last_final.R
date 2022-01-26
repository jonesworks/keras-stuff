library(tidytext)
library(quanteda)
library(quanteda.textstats) 
library(data.table)
library(koRpus)
library(tidyverse)

train <- fread("/Users/a12517/Desktop/feedback-prize-2021/train.csv")

train %>% 
  slice_head(n = 30000) -> train

#feature engineering
flesch <- textstat_readability(train$discourse_text, measure = "Flesch.Kincaid")

train <- cbind(train, flesch)

train %>% 
  select(-document) %>% 
  rename(flesch = `Readability Score`) -> train

# feature engineering
lex_div <- textstat_lexdiv(dfm(train$discourse_text))

lex_div %>% 
  tibble %>% 
  bind_cols(train) -> train

train %>% 
  select(-document) -> train

# feature engineering 
gl <- textstat_readability(train$discourse_text, measure = "Coleman.Liau.grade")

train <- cbind(train, gl)

lex_div %>% 
  tibble %>% 
  bind_cols(train) -> train

train %>% 
  select(-document) -> train

#------feature engineering


train %>% 
  select(-(starts_with("document"))) %>% 
  select(-c(1)) %>% 
  rename(TTR = c(1)) %>% 
  select(-c(2,3,4,5,8,9)) -> train


hapax_proportion <- rowSums(dfm(train$discourse_text) == 1) / ntoken(dfm(train$discourse_text))

hapax_proportion %>% 
  as_tibble() %>% 
  rename(hapax = value) %>% 
  bind_cols(train) -> train

#-------

multicomplaints_train %>% 
  group_by(discourse_type) %>% 
  mutate(wanted = min(hapax*10000),
         wanted = round(wanted, -2)/10000) %>% 
  ungroup() -> multicomplaints_train
