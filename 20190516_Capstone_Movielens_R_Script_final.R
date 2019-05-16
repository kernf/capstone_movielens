# Data Science Capstone Movielens Project - R script
# author: "Florian Kern"
# date: "16 May 2019"
# comment: just the code to obtain the best RMSE value

# load necessary libraries: tidyverse and caret
library(tidyverse)
library(caret)

# run this code that was provided to create the train and test set (edx and validation)

#######################################################
# Create edx set, validation set, and submission file #
#######################################################

# Note: this process could take a couple of minutes
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

# Download the MovieLens 10M dataset:
#  https://grouplens.org/datasets/movielens/10m/
#  http://files.grouplens.org/datasets/movielens/ml-10m.zip
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)
#############################################################

# RMSE is used as a readout, therefore we create a RMSE function:
RMSE <- function(predicted_ratings,true_ratings){
  sqrt(mean((predicted_ratings - true_ratings)^2))
}

# Let's use cross validation to chose lambda.
# penalty value (lambda) sequence
lambdas <- seq(0, 10, 0.25)

# cross validate the lambda sequence
model_6_rmses <- sapply(lambdas, function(l){
  # average rating
  mu <- mean(edx$rating)
  # regularized movie bias
  b_i <- edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  # regularized user bias
  b_u <- edx %>%
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  # predict ratings
  predicted_ratings_6 <-
    validation %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  # return RMSE
  return(RMSE(predicted_ratings_6, validation$rating))
})

# qplot of lambdas vs. RMSES
qplot(lambdas, model_6_rmses)

# determine lambda with the smallest RMSE
lambda <- lambdas[which.min(model_6_rmses)]
lambda

# smallest RMSE value 
final_rmse <- min(model_6_rmses)
final_rmse

# end