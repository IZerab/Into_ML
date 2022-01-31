# this is task 1 of Lorenzo Beltrame submission for Programming Assignment 6 UNIVIE WS2021

# import standard libs
import numpy as np
import pandas as pd


def linear_kernel(a, b, parameter):
    return np.dot(a, np.transpose(b))


def gaussian_kernel(X, X2, kern_param):
    """
    Compute the kernel matrix
    :param kern_param: hyperparameter of the kernel
    :param X: train design matrix
    :param X2: design matrix
    :return: the kernel matrix
    """
    sigma = kern_param
    norm = np.square(np.linalg.norm(X[None, :, :] - X2[:, None, :], axis=2).T)
    return np.exp(-norm / (2 * np.square(sigma)))


def custom_gaussian_kernel(X_train, X, hyperpar):
    """
    Compute the kernel matrix based on a modified RBF kernel
    :param X_train: train design matrix
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
    :param alphas: weight vector for regularized problem
    :param K: kernel matrix
    :param y: target vector
    :return: the value of the cost function
    """
    arg = np.log(1 + np.exp(np.dot(alphas, K)))
    first = np.sum(arg)

    second = -np.dot(y, np.dot(alphas, K))

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

    def fit(self, X, y, gd_step=10, max_steps=100, Lambda=1, epsilon=0.0001,
            max_rate=100, min_rate=0.001):
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

        m = len(X)
        # Save the train in the class
        self.X_train = np.vstack([X.T, np.ones(m)]).T

        # Construct kernel matrix
        K = self.kernel(X, X, self.kern_param)

        # Gradient descent
        self.alphas = np.zeros([m])
        prev_cost = 0
        next_cost = log_reg_cost(K, y, self.alphas, Lambda)
        counter = 0
        cost = 0

        while np.fabs(prev_cost - next_cost) > epsilon:
            neg_grad = - log_reg_gradient(K, y, self.alphas, Lambda)
            best_rate = rate = max_rate
            min_cost = log_reg_cost(K, y, self.alphas, Lambda)
            while rate >= min_rate:
                cost = log_reg_cost(K, y, self.alphas + neg_grad * rate, Lambda)
                if cost < min_cost:
                    min_cost = cost
                    best_rate = rate
                rate /= gd_step
            self.alphas += neg_grad * best_rate
            prev_cost = next_cost
            next_cost = min_cost
            counter += 1
            if counter > max_steps:
                return cost

    def predict_prob(self, X):
        """
        This function predicts the probability associated with a set of instances contained in X (the design matrix)
        :param X: Design matrix
        :return: a list with the probabilities inferred
        """
        # check if the input is a df or not and in case cast it to array
        if isinstance(X, pd.DataFrame):
            X = X.copy().to_numpy()

        X = np.vstack([np.transpose(X), np.ones([len(X)])]).T
        K = self.kernel(self.X_train, X, self.kern_param)
        return sigmoid(np.dot(self.alphas, K))

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

    def decide_chimney(self, X, cost_matrix):
        """
        This function decides which is the best bayesian choice for each entry of the design matrix.
        In thi specific case the cost matrix must be 2 x 2!!
        :param X: design matrix
        :param cost_matrix: matrix (n x n) contining the costs for each action
        :return: the optimal bayesian decisions costs and the list of the action performed
        """
        # get the probabilities
        Z = self.predict_prob(X)

        cost_delivery = []
        cost_not_delivery = []
        decisions = []
        decisions_costs = []

        for i in range(len(Z)):
            # expected cost for the delivery
            cost_delivery.append(cost_matrix[0, 0] * Z[i] + cost_matrix[0, 1] * (1- Z[i]))

            # expected cost for not delivering
            cost_not_delivery.append(cost_matrix[1, 0] * Z[i] + cost_matrix[1, 1] * (1- Z[i]))

            # decide which is the minimum cost
            if cost_delivery[i] < cost_not_delivery[i]:
                decisions_costs.append(cost_delivery[i])
                decisions.append("Delivery")
            else:
                decisions_costs.append(cost_not_delivery[i])
                decisions.append("Do not delivery")

        return decisions_costs, decisions


def decision_chimney_kids(X,log_kids, log_chimneys, cost_vector):
    """
    This function computer the bayesian best decision for Santa. The logistic repressors must have a predict_prob
    module.
    :param log_kids: logistic regressor trained to get if there are kids in the house
    :param log_chimneys: logistic regressor trained to get if there are chimneys in the house
    :param cost_vector: cost of any possible combination, see the PA6 sheet for explaination
    :return: the optimal bayesian decisions costs and the list of the action performed
    """
    decisions = []
    decisions_costs = []

    # prediction whether a house has kids or not
    Z_k = log_kids.predict_prob(X)
    # prediction whether the house has a chimney or not
    Z_c = log_chimneys.predict_prob(X)

    for i in range(len(X)):
        # expected costs for the delivery
        delivery0 = cost_vector[0] * Z_c[i] * Z_k[i]
        delivery1 = cost_vector[1] * Z_c[i] * (1 - Z_k[i])
        delivery2 = cost_vector[2] * (1- Z_c[i]) * Z_k[i]
        delivery3 = cost_vector[3] * (1- Z_c[i]) * (1 - Z_k[i])
        cost_delivery = delivery0 + delivery1 + delivery2 + delivery3

        # expected costs not to deliver
        not_delivery0 = cost_vector[4] * Z_c[i] * Z_k[i]
        not_delivery1 = cost_vector[5] * Z_c[i] * (1 - Z_k[i])
        not_delivery2 = cost_vector[6] * (1 - Z_c[i]) * Z_k[i]
        not_delivery3 = cost_vector[7] * (1 - Z_c[i]) * (1 - Z_k[i])
        cost_not_delivery = not_delivery0 + not_delivery1 + not_delivery2 + not_delivery3

        # decide which is the minimum cost
        if cost_delivery < cost_not_delivery:
            decisions_costs.append(cost_delivery)
            decisions.append("Delivery")
        else:
            decisions_costs.append(cost_not_delivery)
            decisions.append("Do not delivery")

    return decisions_costs, decisions


