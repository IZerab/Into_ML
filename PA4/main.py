# This is the main of the project Programming assignment 4
# author: Lorenzo Beltrame

import matplotlib.pyplot as plt
# lib
import numpy as np
import h5py
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

# custom lib
from Fuctions import add_year
from Fuctions import binarize_rating
from Fuctions import create_dataframe
from Fuctions import custom_train_test
from Fuctions import do_cv
from Fuctions import import_movie_data
from Fuctions import manual_precision_recall
from Fuctions import remove_users
from Fuctions import custom_confusion_matrix
from Fuctions import plot_confusion_matrix




# TASK 1.1
# import the data
df_movies, df_ratings, df_users = import_movie_data()

# create the ML dataframe
df = create_dataframe(df_movies, df_ratings, df_users)

# drop not relevant columns
df.drop(columns=["Timestamp", "Zip-code"], inplace=True)

# remove the users that did less than 200 reviews
df = remove_users(df, 200)

# add the year column and delete the year from the title
df = add_year(df)

# one hot encode the film title, genres and gender
# I used two different encoders to being able to reverse the transformation
my_enc_title = LabelEncoder()

df["Title"] = my_enc_title.fit_transform(df["Title"])

my_enc_genres = LabelEncoder()
df["Genres"] = my_enc_genres.fit_transform(df["Genres"])

my_enc_gender = LabelEncoder()
df["Gender"] = my_enc_genres.fit_transform(df["Gender"])

# binarize the rating ([1,..,5] -> [0,1])
# df_multi is the multilable dataset
df, df_multi = binarize_rating(df)

# drop NaN values
df.dropna(inplace=True)
df_multi.dropna(inplace=True)

# print the columns of the design matrix
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

# subtask 1.2

# number of trials for C hyper parameter
N = 2
# create a log spaced array of 6 elements
x = np.logspace(0.1, 7, N, endpoint=True) / 100
# create the parameter grid
param_grid_SVC = {"C": x}

# I manually insert the values for the MLP classifier (NB: I do not want to have too many hidden layers)
param_grid_MLP = {"hidden_layer_sizes": [(10, 20, 10, 20, 10), (20, 20, 20)]}

# cross validation hyperparameter selection
# SVC
best_score_SVC, best_hyper_SVC, my_SVC = do_cv(
    alg=LinearSVC(max_iter=2000),
    CV_parameters=param_grid_SVC,
    X_train=X_train,
    y_train=y_train,
    cv=3
)

# MLP
print("I tried the following number of")
best_score_MLP, best_hyper_MLP, my_MLP = do_cv(
    alg=MLPClassifier(),
    CV_parameters=param_grid_MLP,
    X_train=X_train,
    y_train=y_train,
    cv=3
)
exit()
# subtask 1.3
# get the predicted probabilities, NB: gridsearch_CV enables the prediction of the probabilities!!
y_prob_svc = my_SVC.predict(X_test)
y_prob_MLP = my_MLP.predict(X_test)

# use the custom plot precision recall curve
precisions_svc, recalls_svc, AP_svc = manual_precision_recall(y_test, y_prob_svc)
print("The Average precision of the SVC is {}".format(AP_svc))
precisions_MLP, recalls_MLP, AP_MLP = manual_precision_recall(y_test, y_prob_MLP)
print("The Average precision of the MLP is {}".format(AP_MLP))

# Print the results
fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(recalls_svc, precisions_svc, label="Support Vector Classifies")
ax.plot(recalls_MLP, precisions_MLP, label="MLP")

# I add the baseline of our model!
baseline = len(y_test[y_test == 1]) / len(y_test)
ax.plot([0, 1], [baseline, baseline], label='Baseline of the model')

ax.set_ylabel('Precision')
ax.set_xlabel('Recall')
ax.legend()
plt.show(block=False)


# subtask 1.4
# train test split
train, test = custom_train_test(df_multi)
print("Multilabel problem")
print("The train set has {} elements.".format(train.shape[0]))
print("The test set has {} elements.".format(test.shape[0]))

# choose the target
y_train = train["Rating"]
y_test = test["Rating"]

# create the design matrix
X_train = train.drop(columns="Rating")
X_test = test.drop(columns="Rating")

# use a SVC
multi_svc = SVC()

# train
multi_svc.fit(X_train, y_train)

# predict
prediction_multi = multi_svc.predict(X_test)

# compute the confusion matrix
my_confusion, my_classes = custom_confusion_matrix(y_test, prediction_multi)

# plot the confusion matrix
plot_confusion_matrix(my_confusion, my_classes)


# Task 2

# subtask 1
# import the data
hf = h5py.File("regression.h5", "r")
X_train = np.array(hf.get("x train"))
y_train = np.array(hf.get("y train"))
X_test = np.array(hf.get("x test"))
y_test = np.array(hf.get("y test"))
hf.close()

# fit the min max scaler
# unite the dataset
temp = np.concatenate((X_train,X_test), axis=0)
my_scaler = MinMaxScaler()
my_scaler.fit(temp)

# apply the normalization
X_train = my_scaler.transform(X_train)
X_test = my_scaler.transform(X_test)


# compute the statistics
# number of instances
num_inst_train = X_train.shape[0]
print("The number of instances in the training dataset is {}".format(num_inst_train))
num_inst_test = X_test.shape[0]
print("The number of instances in the test dataset is {}".format(num_inst_test))

# number of features
num_feat_train = X_train.shape[1]
print("The number of features in the training dataset is {}".format(num_feat_train))
num_feat_test = X_test.shape[1]
print("The number of features in the test dataset is {}".format(num_feat_test))

# list of targets
# I use y_train since it is a bigger dataset!
targets = np.unique(y_train)
print("The targets are {}".format(targets))
num_target = len(targets)
print("The number of targets is {}".format(num_target))


# subtask 2.2


