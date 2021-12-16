# This is the main of the project Programming assignment 4
# author: Lorenzo Beltrame

# lib
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

# custom lib
from Fuctions import import_movie_data
from Fuctions import create_dataframe
from Fuctions import binarize_rating
from Fuctions import remove_users
from Fuctions import custom_train_test
from Fuctions import add_year

# TASK 1.1
# import the data
df_movies, df_ratings, df_users = import_movie_data()

# create the ML dataframe
df = create_dataframe(df_movies, df_ratings, df_users)
# drop NaN
df.drop(columns=["Timestamp", "Zip-code"], inplace=True)
df.dropna(inplace=True)

# remove the users that did less than 200 reviews
df = remove_users(df, 200)

# binarize the rating ([1,..,5] -> [0,1])
df = binarize_rating(df)

# add the year column and delete the year from the title
df = add_year(df)

# print the columsn of the design matrix
print(df.columns.values)

# train test split
train, test = custom_train_test(df)
print("The train set has {} elements.".format(train.shape[0]))
print("The test set has {} elements.".format(test.shape[0]))

# choose the target
y_train = train["Rating"]
y_test = test["Rating"]

# create the design matrix
X_train = train.drop(columns="Rating")
X_test = test.drop(columns="Rating")


