# this is the main of my project
# Author: Lorenzo Beltrame - 23-11-2021

# standard libraries
import numpy as np
import pandas as pd
import sklearn as sk
import random

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer



# custom libraries
from Functions import forward_F_S
from Functions import print_best_scores
# TASK 1

# PREPROCESSING
# load the data
# note that breast_cancer data set has only 2 classes!
X, y = load_breast_cancer(return_X_y=True, as_frame=True)
feat_names = X.columns

# normalize it
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# shuffle the data frame and reset the indexes
random.shuffle(X_scaled)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# create a pandas DF/Series (and adding their columns names train/test gives back a list)
X_train = pd.DataFrame(X_train, columns=feat_names)
X_test = pd.DataFrame(X_test, columns=feat_names)
y_train = pd.Series(y_train)
y_test = pd.Series(y_test)

best_scores, s = forward_F_S(X_train, y_train)
# I print the names of the features, in the order they were selected
print([s,best_scores])
# I plot the scores obtained during feat selection
print_best_scores(best_scores)
print_best_scores(best_scores, log=True)

