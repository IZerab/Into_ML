# this is the lib for the functions
# Author: Lorenzo Beltrame - 23-11-2021

# standard libraries
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import cross_validate as CV

from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt


def forward_F_S(X_train, y_train):
    """
    Function that performs a Greedy Forward feature selection wrt CV SVC. We are using default parameters!!
    :param X_train: Design matrix passed as a pandas DF
    :param y_train: target feature
    :return: CV scores at each iteration, feature list (in order of selection)
    """

    # check if the X input is a pandas DF
    if not isinstance(X_train, pd.DataFrame):
        raise ValueError("Value error: The X input is not a pandas DF!")

    # best features
    s = []
    best_scores = []
    # convenience variables
    features = X_train.columns.to_list()
    remaining_feat = features
    best_scores.append(0)
    iteration = 0

    while len(remaining_feat) > 0:

        # to have a reference
        print("This is iteration {}".format(iteration + 1))

        # initialize scores for this iteration
        scores = []

        for i in range(len(remaining_feat)):
            # I can consider the union
            temp_feat = [remaining_feat[i]] + s
            array_scores = CV(
                estimator=LinearSVC(random_state=42),
                X=X_train[temp_feat],
                y=y_train,
                scoring="f1",
                cv=10)
            scores.append(np.mean(array_scores["test_score"]))

        # check if the new features would do better!
        if all(i < best_scores[iteration] for i in scores):  # to check!!
            break
        # append the best score given by the CV on the previous selecte set of features and the new one
        best_scores.append(max(scores))
        # get its index
        index_max = max(range(len(scores)), key=scores.__getitem__)
        # removing the index
        new_best_feat = remaining_feat.pop(index_max)
        # save the list of best features added in order
        s.append(new_best_feat)
        iteration += 1

    return best_scores, s


def print_best_scores(best_scores, log=False):
    """
    This function prints the best_scores given by the feature selection wrt a cross validated SVC
    :param best_scores: results of the feature selection. If log=True, gives the log scale on y axis.
    :return: graph best scores VS iterations
    """
    # support variable
    x_axis = range(1, len(best_scores))
    best_scores = best_scores[1:len(best_scores)]
    if log:
        plt.plot(x_axis, np.log(best_scores), marker="o")
        plt.title("Best features selected - Log scale", fontsize=20)
        plt.ylabel("Logarithmic Score obtained by the features")
    else:
        plt.plot(x_axis, best_scores, marker="o")
        plt.title("Best features selected", fontsize=20)
        plt.ylabel("Score obtained by the features")
    plt.grid()
    plt.xlabel("Iteration")
    plt.show()


def backward_F_S(X_train, y_train):
    """
    Function that performs a Greedy backward feature selection wrt CV SVC. We are using default parameters!!
    :param X_train: Design matrix passed as a pandas DF
    :param y_train: target feature
    :return: CV scores at each iteration, removed feature list (in order of selection), remaining feat
    """
    # check if the X input is a pandas DF
    if not isinstance(X_train, pd.DataFrame):
        raise ValueError("Value error: The X input is not a pandas DF!")

    # features we are using in the next iterations
    features = X_train.columns.to_list()
    s = features
    best_scores = []
    # convenience variables
    eliminated_feat = []
    # I initialize the first element to be null (for the first comparison)
    best_scores.append(0)
    iteration = 0

    while len(s) > 0:

        # to have a reference
        print("This is iteration {}".format(iteration + 1))

        # initialize scores for this iteration
        scores = []
        for i in range(len(s)):
            # I can consider the difference between sets
            temp_feat = s.copy()
            temp_feat.pop(i)
            array_scores = CV(
                estimator=LinearSVC(random_state=42),
                X=X_train[temp_feat],
                y=y_train,
                scoring="f1",
                cv=10)
            scores.append(np.mean(array_scores["test_score"]))

        # check if the new set without one feature would do better!
        if all(i < best_scores[iteration] for i in scores):  # to check!!
            break
        # append the best score given by the CV on the previous selecte set of features and the new one
        best_scores.append(max(scores))
        # get its index
        index_max = max(range(len(scores)), key=scores.__getitem__)
        # removing the index
        new_worst_feat = s.pop(index_max)
        # save the list of best features added in order
        eliminated_feat.append(new_worst_feat)
        iteration += 1

    return best_scores, eliminated_feat, s
