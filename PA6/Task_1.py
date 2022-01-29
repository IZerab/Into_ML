# this is task 1 of Lorenzo Beltrame submission for Programming Assignment 6 UNIVIE WS2021

# import standard libs
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import pandas as pd


def linear_kernel(a, b, parameter):
    return np.dot(a, np.transpose(b))


def gaussian_kernel(X_train, X, hyperpar):
    """
    Compute the kernel matrix
    :param X: design matrix
    :param hyperpar: normalization factor in the RBF function
    :return: the kernel matrix
    """
    # get the norm of each element of X
    X_norm = np.sum(X ** 2, axis=-1)
    X_train_norm = np.sum(X_train ** 2, axis=-1)

    # get the kernel (decompose the norm in simpler operations!
    sum_norms = X_norm[:, np.newaxis] + X_train_norm[np.newaxis, :]
    K = np.exp((- sum_norms - 2 * np.dot(X, X_train.T)) / (2 * hyperpar ** 2))
    return K


def custom_gaussian_kernel(X_train, X, hyperpar):
    """
    Compute the kernel matrix based on a modified RBF kernel
    :param X: design matrix
    :param hyperpar: tuple, normalization factor in the RBF function and a multiplicative factor
    :return: the kernel matrix
    """
    K1 = gaussian_kernel(X_train[:, :2], X[:, :2], hyperpar[0])
    K2 = hyperpar[2] * gaussian_kernel(X_train[:, 2:], X[:, 2:], hyperpar[1])
    return K1 + K2


def sigmoid(z):
    """
    Computes the sigmoid for a given input
    :param z: float, input of the sigmoid
    :return: the value of the input, it is mapped into [0, 1]
    """
    return np.exp(z) / (1 + np.exp(z))


def log_reg_cost(K, y, alphas, Lambda):
    """
    Cost of the logistic regression
    :param Lambda: regularization parameter
    :param alphas: weight vector for the dual problem
    :param K: kernel matrix
    :param y: target vector
    :return: the value of the cost function
    """
    arg = np.log(1 + np.exp(np.dot(alphas, K)))
    first = np.sum(arg)
    second = -np.dot(y, np.dot(alphas, K))
    # regulariser
    reg = Lambda * alphas.T @ K @ alphas

    return first + second + reg


def log_reg_gradient(K, y, alphas, Lambda):
    """
    Gradient of the loss function
    :param Lambda: regularization parameter
    :param K: kernel matrix
    :param y: target vector
    :param alphas: weight vector for the dual problem
    :return: the gradient of the function (a vector!)
    """
    temp = -np.dot(K, y - sigmoid(np.dot(alphas, K)))
    return temp + Lambda * alphas.T @ (K + K.T)


class custom_log_reg:
    """
    This class implements logistic regression
    """

    def __init__(self, kernel='gaussian', hyper_kernel=None):
        """
        Initialize the function and save the train instances inside the class. Note that for this implementation only
        the gaussian, customized gaussian and the linear and kernels are implemented.
        If a custom kernel is chosen provide a tuple (sigma, var) where sigma is the hyperparameter of the gaussian
        kernel and var is a multiplicative factor.
        """
        # class variables
        self.alphas = None
        self.K = None
        self.X_train = None

        if kernel == "linear":
            self.kernel = linear_kernel
            if hyper_kernel:
                self.kern_param = hyper_kernel
            else:
                self.kern_param = 1
        elif kernel == "gaussian":
            self.kernel = gaussian_kernel
            if hyper_kernel:
                self.kern_param = hyper_kernel
            else:
                self.kern_param = 0.1
        elif kernel == "custom":
            self.kernel = custom_gaussian_kernel
            if hyper_kernel:
                self.kern_param = hyper_kernel
            else:
                self.kern_param = (0.1, 0.1, 1)

    def fit(self, X, y, lr=0.001, max_steps=100, Lambda=1, verbose=False, epsilon=0.001):
        """
        This fuction fits the model to the training data.
        :param max_steps: max number of iterations for the GD
        :param lr: learning rate, a float
        :param Lambda: regularization parameter
        :param verbose: bool, wether or not print the descent of the cost
        :param X: design matrix
        :param y: target vector

        :return:
        """
        # check if the input is a df or not and in case cast it to array
        if isinstance(X, pd.DataFrame):
            X = X.copy().to_numpy()
        if isinstance(y, pd.Series):
            y = y.copy().to_numpy()

        m = X.shape[0]
        # Construct kernel matrix
        self.K = self.kernel(X, X, self.kern_param)
        self.X_train = X

        # Gradient descent
        self.alphas = np.zeros([m])
        costs = [0]

        for j in range(max_steps):
            if j > 500:
                current_lr = lr / j * 100
            else:
                current_lr = lr
            self.alphas -= log_reg_gradient(self.K, y, self.alphas, Lambda=Lambda) * current_lr
            costs.append(log_reg_cost(self.K, y, self.alphas, Lambda=Lambda))

            # stop condition
            if costs[j] - costs[j-1] < epsilon:
                return costs

            if (j % 200 == 0) and verbose:
                print("The cost at the iteration {} is {}".format(j, costs[j]))

        return costs

    def predict_prob(self, X):
        """
        This function predicts the probability associated with a set of instances contained in X (the design matrix)
        :param X: Design matrix
        :return: a list with the probabilities inferred
        """
        # check if the input is a df or not and in case cast it to array
        if isinstance(X, pd.DataFrame):
            X = X.copy().to_numpy()

        return sigmoid(np.dot(self.alphas, self.kernel(X_train=self.X_train, X=X, hyperpar=self.kern_param)))


    def predict(self, X, threshold=0.5):
        """
        This function predicts the labels of a set of instances in a bi-classification problem.
        If the probability is less than the threshold the value 0 is predicted, 1 otherwise
        :param X: The design matrix
        :param threshold: the threshold that is used to infer the label
        :return: a list containing the labels
        """
        # predict the probability
        Z = self.predict_prob(X)

        # give the label
        result = []
        for pred in Z:
            if pred < threshold:
                result.append(0)
            else:
                result.append(1)
        return result

