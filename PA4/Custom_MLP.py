# Author: Lorenzo Beltrame
# this document contains a class that implements a custom Multi layered perceptron
# lib
import numpy as np

# lib from the community
from stochastic_optimizers import AdamOptimizer


# Useful functions
def relu(X):
    """
    Manual implementation of ReLu
    """
    return np.maximum(0, X)


def cost_function(y_pred, y_true):
    """
    Computes the mean squared error
    :param y_pred: predictions vector
    :param y_true: true values targt
    :return: the squared error
    """
    cost = np.sum((y_pred - y_true) ** 2) / len(y_true)
    return cost


def finite_difference(loss_prime, loss_second, y, epsilon):
    """
    THis function computes the finite difference.
    :param loss_prime: loss computed for the primed component (+ epsilon)
    :param loss_second: loss computed for the primed component (- epsilon)
    :param y: true targets
    :param epsilon:
    :return:
    """
    l1 = cost_function(loss_prime, y)
    l2 = cost_function(loss_second, y)
    return (l1 - l2) / 2 * epsilon


def linear_composition(W, X, b):
    """
    Computes the network input as a dot product
    :param W: weight matrix
    :param X: matrix of the instances
    :param b: biases
    :return: weighted sum of the features
    """

    return (X @ W) + b


class my_MLP(object):
    """
    This is a manual implementation of a MLP neural network.
    It contains the fit and predict modules.
    """

    def __init__(self, shape_net, epsilon):
        """
        Initializing function. It creates dictionaries for the weights and the biases, creating an scalable algorithm
        where it is possible to modify the parameters.
        :param shape_net: the shape of the net, i.e. number of instances given when training, number of features, number
                          of neurons in each layer, number of hidden layers, number of outputs
        :param epsilon: parameter to compute the gradient via finite difference
        """
        # fix a seed
        np.random.seed(42)

        # parameter to compute the gradient via finite difference
        self.epsilon = epsilon

        # errors for the training
        self.errors = []

        self.weights = []
        self.biases = []
        self.optimizer = None

        # initialize weights
        for i in range(len(shape_net) - 1):
            w = np.random.normal(loc=0.0, scale=(2 / shape_net[i + 1]), size=(shape_net[i], shape_net[i + 1]))
            self.weights.append(w)

        # initialize biases
        for i in range(1, len(shape_net) - 1):
            self.biases.append(np.zeros((1, shape_net[i])))

    def approximated_gradient(self, X, y, ws, bs):
        """
        Function that implements the approximated gradient using the squared error as the loss function.
        It is preset to work with a list of weight matrices and bias.
        :param X: design matrix
        :param y: target vector
        :param ws: list of weight matrices
        :param bs: list of bias vectors
        :return: the approximated gradient vector for weights and for matrices
        """
        # initialize the list of the gradients
        dW = []
        dB = []

        # do it for each weight matrix
        for k in range(len(ws)):
            w = ws[k]
            dw = np.zeros(w.shape)
            # compute finite difference for every element
            for i in range(w.shape[0]):
                for j in range(w.shape[1]):
                    # add epsilon to one component
                    w[i, j] += self.epsilon
                    pred1 = self.predict(X, ws, bs)
                    # subtract epsilon (2x bc we already added it)
                    w[i, j] -= 2 * self.epsilon
                    pred2 = self.predict(X, ws, bs)

                    # save result in separate matrix
                    dw[i, j] = finite_difference(pred1, pred2, y, self.epsilon)
                    # reset w
                    w[i, j] += self.epsilon
            # store weight matrix gradient in list
            dW.append(dw)

        # iterate through each bias vector
        for k in range(len(bs)):
            b = bs[k]
            db = np.zeros(b.shape)
            # compute finite difference for every element
            for i in range(len(b)):
                # add epsilon to one component
                b[i] += self.epsilon
                pred1 = self.predict(X, ws, bs)

                # subtract epsilon
                b[i] -= 2 * self.epsilon
                pred2 = self.predict(X, ws, bs)

                # save result in separate array
                db[i] = finite_difference(pred1, pred2, y, self.epsilon)
                # reset b
                b[i] += self.epsilon
            # store bias gradient in list
            dB.append(db)

        return dW, dB

    def predict(self, X, W, B):
        """
        Gets the prediction of the net (the forward step)
        """
        # initialize design
        design = X
        # perform forward step
        for i in range(len(W) - 1):
            z = np.dot(design, W[i]) + B[i]
            design = relu(z)

        # return output (no activation and bias here)
        return np.dot(design, W[-1])

    def train(self, X_train, y_train, X_test, y_test, gamma=0.06, max_iter=5000):
        """
        This implements the finite difference approximation of the gradient.
        The test dataset is present for convenience, just to plot the gradient descent.
        :param y_test: test target
        :param X_test: test design matrix
        :param max_iter: maximal number of iteration allowed, default 1000
        :param gamma: learning rate, default: 0.0001
        :param X_train: design matrix used when training
        :param y_train: target vector used when training
        :return: train_cost, test_cost and the number of iteration at which finished
        """
        # initialize necessary variables
        train_cost = []
        test_cost = []
        iterations = []
        # just set a high number
        precedent_cost = 100000000

        # initialize the optimizer
        self.optimizer = AdamOptimizer(self.weights + self.biases, gamma, 0.9, 0.999, self.epsilon)

        for i in range(1, max_iter):
            # compute the gradient for each element of the list
            gradients, bias_gradients = self.approximated_gradient(X_train, y_train, self.weights, self.biases)

            # update the weights
            self.optimizer.update_params(self.weights + self.biases, gradients + bias_gradients)

            # compute loss
            if i % 200 == 0:
                # evaluate on training set
                pred = self.predict(X_train, self.weights, self.biases)
                cost = cost_function(pred, y_train)
                print('Iteration number: {} with cost: {}'.format(i, cost))

                # break condition, fix an appropriate threshold

                # compute the metrixs for the test set
                pred_test = self.predict(X_test, self.weights, self.biases)
                mse_test = cost_function(pred_test, y_test)

                if np.abs(precedent_cost - cost) < 0.007:
                    print('The alg converged')
                    break

                # store current results
                precedent_cost = cost
                train_cost.append(cost)
                test_cost.append(mse_test)
                iterations.append(i)

            return train_cost, test_cost, iterations
