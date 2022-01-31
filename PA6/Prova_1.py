import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class LogisticRegression:
    kern_param = 0
    X = np.array([])
    a = np.array([])
    kernel = None

    def __init__(self, kernel='poly', kern_param=None):
        if kernel == 'poly':
            self.kernel = __linear__
            if kern_param:
                self.kern_param = kern_param
            else:
                self.kern_param = 1
        elif kernel == 'gaussian':
            self.kernel = __gaussian__
            if kern_param:
                self.kern_param = kern_param
            else:
                self.kern_param = 0.1
        elif kernel == 'laplace':
            self.kernel = __laplace__
            if kern_param:
                self.kern_param = kern_param
            else:
                self.kern_param = 0.1

    def fit(self, X, y, max_rate=100, min_rate=0.001, gd_step=10, epsilon=0.0001):
        m = len(X)
        self.X = np.vstack([X.T, np.ones(m)]).T
        # Construct kernel matrix
        K = self.kernel(X, X, self.kern_param)
        # Gradient descent
        self.a = np.zeros([m])
        prev_cost = 0
        next_cost = self.__cost__(K, y, self.a)
        while np.fabs(prev_cost-next_cost) > epsilon:
            neg_grad = -self.__gradient__(K, y, self.a)
            best_rate = rate = max_rate
            min_cost = self.__cost__(K, y, self.a)
            while rate >= min_rate:
                cost = self.__cost__(K, y, self.a+neg_grad*rate)
                if cost < min_cost:
                    min_cost = cost
                    best_rate = rate
                rate /= gd_step
            self.a += neg_grad * best_rate
            prev_cost = next_cost
            next_cost = min_cost

    def predict(self, X):
        X = np.vstack([np.transpose(X), np.ones([len(X)])]).T
        return self.__sigmoid__(np.dot(self.a, self.kernel(self.X, X, self.kern_param)))

    # Kernels

    @staticmethod
    def __sigmoid__(X):
        return np.exp(X) / (1 + np.exp(X))

    @staticmethod
    def __cost__(K, y, a):
        # regulariser
        reg = 0.001 * a.T @ K @ a

        return -np.dot(y, np.dot(a, K)) + np.sum(np.log(1 + np.exp(np.dot(a, K)))) + reg

    @classmethod
    def __gradient__(cls, K, y, a):
        return -np.dot(K, y - cls.__sigmoid__(np.dot(a, K))) + 0.001 * a.T @ (K + K.T)

    def predict_label(self, X, threshold=0.5):
        """
        This function predicts the labels of a set of instances in a bi-classification problem.
        If the probability is less than the threshold the value 0 is predicted, 1 otherwise
        :param X: The design matrix
        :param threshold: the threshold that is used to infer the label
        :return: a list containing the labels
        """
        # predict the probability
        Z = self.predict(X)

        # give the label
        result = []
        for pred in Z:
            if pred < threshold:
                result.append(0)
            else:
                result.append(1)

        return result


def __laplace__(a, b, kern_param):
    mat = np.zeros([len(a), len(b)])
    for i in range(0, len(a)):
        for j in range(0, len(b)):
            mat[i][j] = np.exp(-np.linalg.norm(np.subtract(a[i], b[j])) / kern_param)
    return mat


def __gaussian__(a, b, kern_param):
    mat = np.zeros([len(a), len(b)])
    for i in range(0, len(a)):
        for j in range(0, len(b)):
            mat[i][j] = np.exp(-np.sum(np.square(np.subtract(a[i], b[j]))) / (2 * kern_param * kern_param))
    return mat


def __linear__(a, b, parameter):
    return np.dot(a, np.transpose(b))


