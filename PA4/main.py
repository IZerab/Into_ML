# This is the main of the project Programming assignment 4
# author: Lorenzo Beltrame

# lib
import pandas as pd
import numpy as np
import csv
from sklearn.neural_network import MLPClassifier

# custom lib
from Fuctions import import_movie_data
from Fuctions import create_dataframe
from Fuctions import binarize_rating
from Fuctions import remove_users
from Fuctions import custom_train_test

# TASK 1.1
# import the data
df_movies, df_ratings, df_users = import_movie_data()

# create the ML dataframe
df = create_dataframe(df_movies, df_ratings, df_users)
# drop NaN
df.dropna(inplace=True)

# remove the users that did less than 200 reviews
df = remove_users(df, 200)

# binarize the rating ([1,..,5] -> [0,1])
df = binarize_rating(df)

# train test split
train, test = custom_train_test(df)
print("The train set has {} elements.".format(train.shape[0]))
print("The test set has {} elements.".format(test.shape[0]))
