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
from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from joblib import parallel_backend                 # parallelize using CPU!!!!

# custom libraries
from Functions import forward_F_S
from Functions import backward_F_S
from Functions import print_best_scores
from Functions import sparse_data_sample
from Functions import get_param_grid
from Functions import nice_kernel

# fix the seed
random.seed(40)

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

print("Forward Feature selection: ")
best_scores, s_f = forward_F_S(X_train, y_train, break_cond=False)
# I print the names of the features, in the order they were selected
print(s_f, "\n")
# I plot the scores obtained during feat selection
print_best_scores(best_scores)

# subtask 2
print("Backward Feature selection: ")
best_scores, eliminated_feat = backward_F_S(X_train, y_train, break_cond=False)
# I print the eliminated features in the order they were eliminated
print(eliminated_feat, "\n")
# I plot the scores obtained during feat selection
print_best_scores(best_scores)
print("Features chosen in order by the backward FS: \n", eliminated_feat)

# subtask 3
#best 6 features by each argument
print("Forward Feature selection: ", s_f[s_f[index] < 6])
print("Backward Feature selection: ", eliminated_feat[eliminated_feat > len(feat_names) - 7])

exit()


# TASK 2

# define directory for storing data
dir = '/data'

# Loading training data
# note that 20 newsgroup data set data set has 20 classes!
train = fetch_20newsgroups_vectorized(
    data_home=dir, subset='train', remove=("headers", "footers", "quotes"))

X_train = train.data
y_train = train.target

# Loading testing data
test = fetch_20newsgroups_vectorized(
    data_home=dir, subset='test', remove=("headers", "footers", "quotes"))
X_test = test.data
y_test = test.target

# get a subsample of the data, since they are too big (set do_it to false to skip this)
X_train, X_test, y_train, y_test = sparse_data_sample(X_train, X_test, y_train, y_test, 8000, 3000, do_it=True)


"""

# Subtask 1

# subset with the different kernels I want to try
kernel_names = ["linear", "poly", "rbf", "sigmoid"]


# DEFAULT PARAMETER SUPPORT VECTOR CLASSIFIERS

score_train = []
score_test = []
# support variable
iter = 0

for ker in kernel_names:
    # I initialize my learning alg
    my_svc = SVC(kernel=ker)
    # The next line is to parallelize
    with parallel_backend('threading', n_jobs=-1):
        my_svc.fit(X_train, y_train)

    # predictions
    pred_train = my_svc.predict(X_train)
    pred_test = my_svc.predict(X_test)

    # save accuracies
    score_train.append(accuracy_score(pred_train, y_train))
    score_test.append(accuracy_score(pred_test, y_test))

    # print the results
    print("Accuracies for the vanilla {} kernel:    TRAIN: {}    TEST: {} \n".format(
        kernel_names[iter], score_train[iter], score_test[iter]))
    iter += 1


# TUNE HYPERPARAMETERS SUPPORT VECTOR CLASSIFIERS

best_score_train = []
best_score_test = []
best_hyperparameters = []
# support variable
iter = 0

# I perform a Grid Search CV to tune SVC hyperparameters

# as before, the next line is performed for parallelization

for ker in kernel_names:

    # I initialize my learning alg
    my_svc = SVC(kernel=ker)

    # I get my grid
    CV_parameters = get_param_grid(ker)

    # Set up the actual algorithm
    my_grid_CV = GridSearchCV(estimator=my_svc, param_grid=CV_parameters, cv=3, n_jobs=-1)

    # select the model
    # The next line is to parallelize
    with parallel_backend('threading', n_jobs=-1):
        my_grid_CV.fit(X_train, y_train)

    # predictions
    best_pred_train = my_grid_CV.predict(X_train)
    best_pred_test = my_grid_CV.predict(X_test)

    # save scores
    best_score_train.append(accuracy_score(y_train, best_pred_train))
    best_score_test.append(accuracy_score(y_test, best_pred_test))

    # save selecting hyperparameters
    best_hyperparameters.append(my_grid_CV.best_params_)

    # print results
    print("Accuracies for the tuned {} kernel:    TRAIN: {}    TEST: {}".format(
        kernel_names[iter], best_score_train[iter], best_score_test[iter]))
    print("With the following hyperparameters: {} \n".format(best_hyperparameters[iter]))
    iter += 1


"""

# subtask 2

# I initialize my learning alg
my_svc = SVC(kernel=nice_kernel)
# The next line is to parallelize
with parallel_backend('threading', n_jobs=-1):
    my_svc.fit(X_train, y_train)

# predictions
pred_train = my_svc.predict(X_train)
pred_test = my_svc.predict(X_test)

# save accuracies
nice_score_train = (accuracy_score(pred_train, y_train))
nice_score_test = (accuracy_score(pred_test, y_test))

# print the results
print("Accuracies for the nice kernel:    TRAIN: {}    TEST: {} \n".format(
    nice_score_train, nice_score_test))











