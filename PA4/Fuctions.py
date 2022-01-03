# Those are useful functions for the project Programming assignment 4
# author: Lorenzo Beltrame

import itertools
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import parallel_backend
from scipy.special import expit
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


def import_movie_data():
    """
    This function import movie data from the files present in the folder, be sure that they are present!
    Note that the separator of the .dat files is "::" and you need to pass tha names of the columns
    :return: 3 pandas dataframes: df_movies, df_ratings, df_users
    """
    # I manually insert the column names
    col_movies = ["MovieID", "Title", "Genres"]
    col_ratings = ["UserID", "MovieID", "Rating", "Timestamp"]
    col_users = ["UserID", "Gender", "Age", "Occupation", "Zip-code"]

    # read flash.dat to a list of lists
    df_movies = pd.read_table('movies.dat', sep=r"::", header=None, engine="python", names=col_movies)
    df_ratings = pd.read_table('ratings.dat', sep=r"::", header=None, engine="python", names=col_ratings)
    df_users = pd.read_table('users.dat', sep=r"::", header=None, engine="python", names=col_users)
    return df_movies, df_ratings, df_users


def create_dataframe(df_movies, df_ratings, df_users):
    """
    This function performs a series of "Pandas.merge" methods to create a dataframe in which each entry is a review
    with the informations of the related film
    :param df_movies: dataframe containing info about the movies
    :param df_ratings: dataframe containing info about the ratings
    :param df_users: dataframe containing info about the users
    :return: dataframe to be used in the ML
    """
    # check if they are all pandas DF
    inputs = [df_movies, df_ratings, df_users]
    for i in inputs:
        if not isinstance(i, pd.DataFrame):
            raise ValueError("Value error: The input is not a pandas DF!")

    # the base for our joining in df_ratings!
    df_ratings = df_ratings.merge(df_movies, on="MovieID", how="left")
    df = df_ratings.merge(df_users, on="UserID", how="left")

    return df


def remove_users(df, N):
    """
    Remove all users which have rated less than N movies.
    :param df: pandas dataframe that includes movies
    :param N: threshold of movies watched
    :return:
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Value error: The input is not a pandas DF!")

    temp1 = df.shape[0]

    # group by the user feature (they have the same index)
    groups = df.groupby(by="UserID").count()[lambda x: x >= N]
    # I remove the NaN created in the last passage
    groups.dropna(inplace=True)
    # select only the chosen indexes
    df = df[df["UserID"].isin(groups.index)]

    temp2 = df.shape[0]
    print("The number of reviews dropped is: ", temp1 - temp2)

    return df


def binarizer(x):
    """
    Rule to binarize the data. See binarize rating for more
    :param x: convenience variable
    :return: x binarized
    """
    if x == 5 or x == 4:
        # binarize to 1
        x = 1
    else:
        # binarize to 0
        x = 0
    return x


def binarize_rating(df):
    """
    The ratings are given as 1-star to 5-star ratings. Convert these ratings to binary
    labels, such that 4 and 5 stars ratings are mapped to label “1” and 1, 2, and 3 stars
    ratings are mapped to label “0”. We are using the convenience function binarize.
    :param df: Pandas dataframe containing the aggregated movie data
    :return: the dataframe with the binarized columns and the dataframe without binarization
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Value error: The input is not a pandas DF!")

    # I create a copy since we are going to slice!
    old_df = df.copy()
    temp_df = df.copy()
    temp_df["Rating"] = temp_df["Rating"].apply(binarizer)

    return temp_df, old_df


def remove_text_between_parentheses(text):
    """
    Removes the text between the parentheses.
    Credits to Wiktor Stribiżew
    url: https://stackoverflow.com/questions/37528373/how-to-remove-all-text-between-the-outer-parentheses-in-a-string
    :param text: string where we want to eliminate the parentheses
    :return: the string without the parentheses and the text between them
    """
    n = 1  # run at least once
    while n:
        # remove non-nested/flat balanced parts
        text, n = re.subn(r'\([^()]*\)', '', text)
    return text


def find_text_between_parentheses(s):
    """
    Find the text between the parentheses "()" of the given input string.
    I added a sanity check to see if the text was a digit or not!
    :param s: input string
    :return: the text between the parenthese
    """
    if not isinstance(s, str):
        raise ValueError("The input is not a string!")

    result = s[s.find("(") + 1:s.find(")")]
    if result.isdigit():
        return result
    else:
        pass


def add_year(df):
    """
    Function that extract the year from the tile and adds it as a new column value.
    Works only for this program!!!!
    :param df:Pandas dataframe containing the aggregated movie data
    :return: the dataframe with a new year column and a new title column
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Value error: The input is not a pandas DF!")

    # add the years to the DF
    df["Year"] = df["Title"].apply(find_text_between_parentheses)
    # delete the year between the parentheses
    df["Title"] = df["Title"].apply(remove_text_between_parentheses)

    return df


def custom_train_test(df):
    """
    Custom train test split: the test are all the ratings of users with user ids 1, . . . , 1000
    (unless a user was removed). The remaining are the train.
    :param df: Pandas dataframe containing the aggregated movie data
    :return: train and test pandasDF
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Value error: The input is not a pandas DF!")
    temp_df = df.copy()
    train = temp_df[temp_df["UserID"] > 1000]
    test = temp_df[temp_df["UserID"] <= 1000]
    return train, test


def do_cv(alg, CV_parameters, X_train, y_train, cv):
    """
    This function does the cross validation parameter selection.
    :param alg: pass the sklearn alg to use (note: pass the class!)
    :param CV_parameters: dictionary containing the parameter
    :param X_train: train design matrix
    :param y_train: train target
    :param cv: number of folds to use
    :return: the best score, the best hyperparameter and the already fitted model
    """

    # I initialize my learning alg
    my_alg = alg

    # Set up the actual algorithm
    my_grid_CV = GridSearchCV(estimator=my_alg, param_grid=CV_parameters, cv=cv)

    # select the model
    # The next line is to parallelize
    with parallel_backend('threading', n_jobs=-1):
        my_grid_CV.fit(X_train, y_train)

    # predictions
    best_pred_train = my_grid_CV.predict(X_train)
    # save scores
    best_score_train = accuracy_score(y_train, best_pred_train)
    # save selecting hyperparameters
    best_hyperparameters = my_grid_CV.best_params_
    # save the best model
    model = my_grid_CV.best_estimator_

    # print results
    print("Accuracies for the tuned model an the train set: {}".format(best_score_train))
    print("With the following hyperparameters: {} \n".format(best_hyperparameters))

    return best_score_train, best_hyperparameters, model


def precision_recall(y_test, y_pred):
    """
    Function that computes the precision and the recall of a classifier
    :param y_test: true labels for the test dataset
    :param y_pred: prediction given by the design matrix of the test dataset
    :return: precision [0,1], recall [0,1]
    """
    # save for convenience a local copy
    y_test_local = y_test
    y_pred_local = y_pred
    # Initialize the counters
    # true positive
    TP = 0
    # false positive
    FP = 0
    # false negative
    FN = 0

    # Compute manually the confusion matrix
    for i in range(len(y_test_local)):
        # note that we work with manually binarized data given by thresholding!
        if y_test_local[i] == y_pred_local[i] == 1:
            TP += 1
        if y_pred_local[i] == 1 and y_test_local[i] != y_pred_local[i]:
            FP += 1
        if y_pred_local[i] == 0 and y_test_local[i] != y_pred_local[i]:
            FN += 1

    # Calculate precision
    try:
        precision = TP / (TP + FP)
    except:
        precision = 1

    # calculate recall
    try:
        recall = TP / (TP + FN)
    except:
        recall = 1

    return precision, recall


def compute_AP(precision, recall):
    """
    It computes the average precision by computing the area under the precision recall curve by binning.
    :param precision: precision of a given algorithm
    :param recall: recall of a given algorithm
    :return: the Average precision value
    """
    # I add the [0,0] points to make binning easier!
    precision = np.concatenate([[0.0], precision, [0.0]])

    # I center the binning
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = np.maximum(precision[i - 1], precision[i])

    # I find the index where the value changes
    recall = np.concatenate([[0.0], recall, [1.0]])
    changing_points = np.where(recall[1:] != recall[:-1])[0]

    # compute the area of the single bins
    areas = (recall[changing_points + 1] - recall[changing_points]) * precision[changing_points + 1]
    # sum all the bins!
    result = areas.sum()
    return result


def manual_precision_recall(y_test, y_prob):
    """
    Function that computes the precisions, the recalls with different thresholds. I also computes the average precision
    (AP) for the model taken into consideration.
    :param y_test: binary target
    :param y_prob: predicted probabilities related to the test design matrix
    :return: precisions and recalls computed for the different values of the threshold, AP
    """
    # Useful lists
    precisions = []
    recalls = []

    # Define M thresholds to use
    M = 100
    thresholds = np.linspace(0, 1, num=M)

    # Find precision and recall
    for T in thresholds:
        y_test_pred = []

        # manually discriminate given the threshold
        for prob in y_prob:
            if prob > T:
                y_test_pred.append(1)
            else:
                y_test_pred.append(0)

        # call the previously defined function!
        precision, recall = precision_recall(y_test, y_test_pred)

        precisions.append(precision)
        recalls.append(recall)

    # compute the ap
    AP = compute_AP(precision=precisions, recall=recalls)

    return precisions, recalls, AP


def custom_confusion_matrix(y_true, y_pred):
    """
    Computes the confusion matrix of any multilabelled problem
    :param y_true: true target of the test data
    :param y_pred: predicted target from the test confusion matrix
    :return: the confusion matrix and the classes
    """
    # get the values of the different classes
    my_classes = np.unique(y_true)

    # allocate memory for the confusion matrix
    supp_mat = np.zeros((len(my_classes), len(my_classes)))

    # manually count the correspondences between actual and predicted classes
    for i in range(len(my_classes)):
        for j in range(len(my_classes)):
            supp_mat[i, j] = np.sum((y_true == my_classes[i]) & (y_pred == my_classes[j]))

    return supp_mat, my_classes


def plot_confusion_matrix(confusion, classes):
    """
    This function prints and plots the confusion matrix.
    """
    plt.imshow(confusion, interpolation='nearest')
    plt.title("Confusion Matrix for the multilable problem")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = confusion.max() / 2.
    for i, j in itertools.product(range(confusion.shape[0]), range(confusion.shape[1])):
        plt.text(j, i, format(confusion[i, j]),
                 horizontalalignment="center",
                 color="white" if confusion[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def custom_MLP():
    z = np.arange(-5.0, 5.0, 0.1)
    a = expit(z)
    df = pd.DataFrame({"a": a, "z": z})
    df["z1"] = 0
    df["a1"] = 0.5
    sigmoid = alt.Chart(df).mark_line().encode(x="z", y="a")
    threshold = alt.Chart(df).mark_rule(color="red").encode(x="z1", y="a1")
    (sigmoid + threshold).properties(title='Chart 1')
