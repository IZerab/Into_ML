# this is the lib for the functions
# Author: Lorenzo Beltrame - 23-11-2021

# standard libraries
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import cross_validate as CV
import random
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import scipy

# set random seed
random.seed(40)

def forward_F_S(X_train, y_train, break_cond = False):
    """
    Function that performs a Greedy Forward feature selection wrt CV SVC. We are using default parameters!!
    :param break_cond: if true, breaks if all the new subsets are worse than last one
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
        if iteration % 10 == 0:
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
        if all(i < best_scores[iteration] for i in scores) and break_cond:
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
    :param log : if true plots with a logaritmic scale on Y
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


def backward_F_S(X_train, y_train, break_cond=False):
    """
    Function that performs a Greedy backward feature selection wrt CV SVC. We are using default parameters!!
    :param X_train: Design matrix passed as a pandas DF
    :param y_train: target feature
    :param break_cond: if true, breaks if all the new subsets are worse than last one
    :return: CV scores at each iteration, removed feature list (in order of selection), remaining feat (This last one
                only if break_cond=True)
    """
    # check if the X input is a pandas DF
    if not isinstance(X_train, pd.DataFrame):
        raise ValueError("Value error: The X input is not a pandas DF!")

    # features we are using in the next iterations
    features = X_train.columns.to_list()
    s = features
    num_feat = len(features)
    best_scores = []
    # convenience variables
    eliminated_feat = []
    # I initialize the first element to be null (for the first comparison)
    best_scores.append(0)
    iteration = 0

    while iteration < num_feat:
        # to have a reference
        if iteration % 5 == 0:
            print("This is iteration {}".format(iteration))

        # initialize scores for this iteration
        scores = []
        for i in range(len(s)):
            # small condition to avoid error on last iteration
            if iteration == num_feat - 1:
                temp_feat = s.copy()
            else:
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
        if all(i < best_scores[iteration] for i in scores) and break_cond:
            break
        # append the best score given by the CV on the previous selected set of features and the new one
        best_scores.append(max(scores))
        # get its index
        index_max = max(range(len(scores)), key=scores.__getitem__)
        # removing the index
        new_worst_feat = s.pop(index_max)
        # save the list of best features added in order
        eliminated_feat.append(new_worst_feat)
        iteration += 1

    if break_cond:
        return best_scores, eliminated_feat, s
    else:
        return best_scores, eliminated_feat


def sparse_data_sample(X_train, X_test, y_train, y_test, sub_size_train, sub_size_test, do_it=True):
    """
    get a subsample otherwise the data are too big :(((
    :param sub_size_train: size of the resulting sample for the train part
    :param sub_size_test: size of the resulting sample for the test part
    :type do_it: if True samples the data, if not does not (useful to see the different results when reviewing the code)
    :return: X_train, X_test, y_train, y_test of the correct size
    """
    if do_it:
        i_train = np.random.choice(np.arange(X_train.shape[0]), sub_size_train, replace=False)
        i_test = np.random.choice(np.arange(X_test.shape[0]), sub_size_test, replace=False)

        X_train = X_train[i_train]
        X_test = X_test[i_test]
        y_train = y_train[i_train]
        y_test = y_test[i_test]

        return X_train, X_test, y_train, y_test
    else:
        return X_train, X_test, y_train, y_test


def get_param_grid(kernel):
    """
    This function gets as an argument a kernel and gives back the param grid to be used in grid search CV
    :param kernel: name of the kernel ["linear", "poly", "rbf", "sigmoid"]
    :return: the grid for that specific kernel
    """

    if not kernel in ["linear", "poly", "rbf", "sigmoid"]:
        raise ValueError("Insert the keyword of a valid kernel!")

    grid = []

    if kernel == "linear":
        # for the linear kernel coef0, degree and gamma are ignored params!!
        grid = {"decision_function_shape": ["ovo", "ovr"]}

    if kernel == "poly":
        grid = {'coef0': [0, 0.6, 1, 2],
                'degree': [2, 3, 4, 6],
                'gamma': [0.1, 0.6, 1, 1.3, "scale", "auto"]}

    if kernel == "rbf":
        grid = {'gamma': [0.1, 0.6, 1, 1.6, 3, 5, "scale", "auto"]}

    if kernel == "sigmoid":
        grid = {'coef0': [0, 0.6, 1, 1.6, 2, 5, 10],
                'gamma': [0.1, 0.6, 1, 1.6, 3, 5, "scale", "auto"]}

    return grid


def nice_kernel(x, y):
    """
    Kernel to be used in the learning alg. It has to be passed into the SVC's "kernel" parameter!!
    I manually set a multiplicative hyper parameter that multiplies the tanget of the dot product between X and X'.
    :param x:  the design matrix of what we want to predict
    :param y:  the train design matrix that is passed to the SVC
    :return: the evaluated kernel function to be passed to the SVC
    """
    # compute the dot product between X' and wic
    result = x.dot(y.transpose())
    result = result.tan()
    # include a multiplicative hyper parameter (I did not call it in the arguments for convenience) I
    C = 1.4
    result = result.multiply(C)
    # compute the hadamard product between the previous matrix and itself!!
    return result
