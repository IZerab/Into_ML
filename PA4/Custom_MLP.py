# Author: Lorenzo Beltrame
# this document contains a class that implements a custom Multi layered perceptron
import numpy as np


# Useful functions
def relu(array):
    """
    Manual implementation of ReLu
    :param array: numerical array
    :return: array on which we computed the relu
    """
    array[array < 0] = 0
    return array

def cost_function(A, y_true):
    """
    Computes the squared error
    :param A: neuron activation
    :param y_true: true values targt
    :return: the squared error
    """
    return (np.mean(np.power(A - y_true, 2))) / 2


def linear_composition(W, X, b):
    """
    Computes the network input as a dot product
    :param W: weight matrix
    :param X: matrix of the instances
    :param b: biases
    :return: weighted sum of the features
    """

    return (X @ W) + b


class my_MLP():
    def __init__(self, n_features, n_neurons, n_layers, n_output):
        """
        Initializing function. It creates dictionaries for the weights and the biases, creating an scalable algorithm
        where it is possible to modify the parameters.
        :param n_features: number of features
        :param n_neurons: number of neurons in each layer
        :param n_layers: number of layers
        :param n_output: number of outputs
        """
        # fix a seed
        np.random.seed(42)

        # save the parameters
        self.n_features = n_features
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        self.n_output = n_output

        # create a dictionary for the Z and for the results of the activation functions
        self.Z = {}
        self.S = {}

        # errors for the training
        self.errors = []

        # initialize biases and weights
        self.weights = {}
        self.biases = {}
        for i in range(n_layers):
            self.weights[i] = np.random.uniform(size=(n_features, n_neurons)
            self.biases[i] = np.zeros(size=(1, n_neurons)

    def predict(self, X):
        """
        computes predictions with learned parameters
        :param X: design matrix
        :return: a vector containing the predictions
        """
        for i in range(self.n_layers):
            # linear combination
            self.Z[i] = linear_composition(self.weights[i], X, self.biases[i])
            self.S[i] = relu(Z[i])

        return np.where(S2 >= 0.5, 1, 0)

    def fit_forward(self, X, y, n_features=2, n_neurons=3, n_output=1, iter, eta=0.001):
        """
        Performs a forward propagation.
        :param X: design matrix
        :param y: target vector
        :param n_features: number of features
        :param n_neurons: number of neurons in each hidden layer
        :param n_output: number of output neurons
        :param iter: number of iterations of the alg
        :param gamma: learning rate
        :return: errors over iterations, a dictionary of the learned parameters
        """
        for _ in range(iter):
            for i in range(self.n_layers):
                self.Z[i] = linear_combinations(self.weights[i], X, self.biases[i])
                self.S[i] = relu(self.Z[i])

        # I compute the errors
            self.errors = cost_function(S2, y)
            errors.append(error)

            ##~~ Backpropagation ~~##

            # update output weights
            delta2 = (S2 - y) * S2 * (1 - S2)
            W2_gradients = S1.T @ delta2
            param["W2"] = param["W2"] - W2_gradients * eta

            # update output bias
            param["b2"] = param["b2"] - np.sum(delta2, axis=0, keepdims=True) * eta

            # update hidden weights
            delta1 = (delta2 @ param["W2"].T) * S1 * (1 - S1)
            W1_gradients = X.T @ delta1
            param["W1"] = param["W1"] - W1_gradients * eta

            # update hidden bias
            param["b1"] = param["b1"] - np.sum(delta1, axis=0, keepdims=True) * eta

        return errors, param