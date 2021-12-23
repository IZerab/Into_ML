# Author: Lorenzo Beltrame
# this document contains a class that implements a custom Multi layered perceptron
# lib
import numpy as np

# lib from the community
from stochastic_optimizers import AdamOptimizer


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


def approximated_gradient(X, y, W, B, epsilon):
    """
    Function that implements the approximated gradient using the squared error as the loss function.
    It is preset to work with a list of weight matrices and bias.
    :param X: design matrix
    :param y: target vector
    :param W: list of weight matrices
    :param B: list of bias vectors
    :param epsilon: parameter to compute the gradient via finite difference
    :return: the approximated gradient vector
    """
    # list of gradients
    weight_gradient = []
    bias_gradient = []

    for k in range(len(W)):
        w = W[k]
        dw = np.zeros(w.shape)
        # compute the finite difference
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                # add epsilon to one component
                w[i, j] += epsilon
                pred1 = self.forward(X, w, bs)
                # subtract epsilon (2x bc we already added it)
                w[i, j] -= 2 * epsilon
                pred2 = model.forward(X, ws, bs)

                # save result in separate matrix
                dw[i, j] = self.finite_difference(pred1, pred2, y, epsilon)
                # reset w
                w[i, j] += epsilon
        # store weight matrix gradient in list
        dW.append(dw)

    # iterate through each bias vector
    for k in range(len(bs)):
        b = bs[k]
        db = np.zeros(b.shape)
        # compute finite difference for every element
        for i in range(len(b)):
            # add epsilon to one component
            b[i] += epsilon
            pred1 = model.forward(X, ws, bs)

            # subtract epsilon
            b[i] -= 2 * epsilon
            pred2 = model.forward(X, ws, bs)

            # save result in separate array
            db[i] = self.finite_difference(pred1, pred2, y, epsilon)
            # reset b
            b[i] += epsilon
        # store bias gradient in list
        dB.append(db)

    return dW, dB


class my_MLP:
    """
    This is a manual implementation of a MLP neural network.
    It contains the fit and predict modules.
    """

    def __init__(self, n_features, n_neurons, n_layers, n_output, epsilon):
        """
        Initializing function. It creates dictionaries for the weights and the biases, creating an scalable algorithm
        where it is possible to modify the parameters.
        :param n_features: number of features
        :param n_neurons: number of neurons in each layer
        :param n_layers: number of layers
        :param n_output: number of outputs
        :param epsilon: parameter to compute the gradient via finite difference
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

        # parameter to compute the gradient via finite difference
        self.epsilon = epsilon

        # errors for the training
        self.errors = []

        # initialize biases and weights
        self.weights = {}
        self.biases = {}
        for i in range(n_layers):
            # sampling randomly for weights
            self.weights[i] = np.random.uniform(size=(n_features, n_neurons)
            # setting all biases to 0
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
            self.S[i] = relu(self.Z[i])

        #################### chack ############################################################

        return np.where(S2 >= 0.5, 1, 0)

    def forward(self, X, y):
        """
        Performs a forward propagation.
        :param X: design matrix
        :param y: target vector
        :param itera: number of iterations of the alg
        :return: errors over iterations, a dictionary of the learned parameters
        """
        for i in range(self.n_layers):
            self.Z[i] = linear_composition(self.weights[i], X, self.biases[i])
            if i != self.n_layers - 1:
                self.S[i] = relu(self.Z[i])
            else:
                self.S[i] = self.Z[i]
        # I compute the errors
        error = cost_function(self.S[self.n_layers], y)
        self.errors.append(error)

    def train(self, X, y, gamma=0.0001):
        """
        This implements the finite difference approximation of the gradient.
        :param gamma: learning rate, default: 0.0001
        :param X: design matrix
        :param y: target vector
        :return:
        """
        # get the lists from the dictionaries
        weights = list(self.weights.values())
        biases = list(self.biases.values())
        gradients = []
        bias_gradients = []
        params = np.concatenate((weights, biases))

        # initialize the optimizer
        optimizer = AdamOptimizer(
            params,
            learning_rate_init=gamma,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-06)

        # compute the gradient for each element of the list
        for i in range(len(weights)):
            gradients.append(approximated_gradient(X=X, y=y, W=weights, B=biases, epsilon=1e-06))
            bias_gradients.append(approximated_gradient(X=X, y=y, W=weights, B=biases, epsilon=1e-06, bias_case=True))

        # concatenate the gradients
        param_gradients = np.concatenate((gradients, bias_gradients))
        # update the weights

        optimizer.update_params(param_gradients)

