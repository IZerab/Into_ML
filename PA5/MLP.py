# This is the core implementation done by professor Sebastian Tschiatschek @ UNIVIE
# This file contains a useful

import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn import preprocessing
from tqdm import tqdm
import h5py
import time

from _stochastic_optimizers import AdamOptimizer


def square(t, gradient=False):
    """
    This is the activation function that might be used by the learning alg
    :param gradient: bool, flag used to indicate whether the gradient of the activation should be computed.
    :param t: pre-activation, this is an array!
    :return: activations after the activation function and its gradient (if Gradient==True)
    """
    activation = np.square(t)
    if not gradient:
        return activation
    else:
        return activation, [ent * 2 for ent in t]


def relu(t, gradient=False):
    """Rectified linear unit activation function.

    Parameters
    ----------
    t : numpy array
        pre-activations
    gradient : bool, optional
        A flag used to indicate whether the gradient of the activation should
        be computed.

    Returns
    -------
    activations : numpy array
        output after applying the activation function
    gradient : numpy array
        gradient of activation function. Only returned if flag is set to true.
    """
    t = np.maximum(0, t)
    if not gradient:
        return t
    else:
        return t, (t > 0).astype(float)


def identity(t, gradient=False):
    """Identity activation function.

    Parameters
    ----------
    t : numpy array
        pre-activations
    gradient : bool, optional
        A flag used to indicate whether the gradient of the activation should
        be computed.

    Returns
    -------
    activations : numpy array
        output after applying the activation function
    gradient : numpy array
        gradient of activation function. Only returned if flag is set to true.
    """
    if not gradient:
        return t
    else:
        return t, np.ones_like(t)


class MyMLP:

    def __init__(self, hidden_layer_sizes=(10, 10,), activations=None):
        """
        :param hidden_layer_sizes: Size of the net (insert a tuple of values)
        :param activations: activation function to be used, pass a function (Relu, sigma, identity) specifying the list
                            used, one activation function for each layer
        """
        if activations is None:
            activations = [square, relu, identity]
        assert len(activations) == len(hidden_layer_sizes), "Invalid number of layers/activations."

        self.hidden_layer_sizes = list(hidden_layer_sizes)
        # single output at final layer
        self.hidden_layer_sizes.append(1)
        self.activations = activations
        # the last layer has the identity as activation function!
        self.activations.append(identity)

        # weights and biases
        self.weights = []
        self.biases = []

        self.lr = 1e-3  # learning rate
        self.alpha = 1e-5  # weight decay

        # caches for post activations and derivatives of activation functions
        self.cache_post_activations = []
        self.cache_derivatives = []

    def fit(self, x, y, x_test=None, y_test=None, bs=10, n_epochs=500, method='backprop'):
        """
        Optimize the parameters of the neural network to minimze the mse on the
        training data.

        Parameters
        ----------
        x : numpy array
            Input data (shape n_samples x n_features)
        y : numpy array
            Targets (shape n_samples x 1)
        x_test : numpy array, optional
            Test data inputs (to track performance)
        y_test : numpy array, optional
            Test data targets (to track performance)
        bs : int, optional
            batch size for mini-batch sgd
        n_epochs : int, optional
            number of epochs for optimization
        method : int, optional
            method for computing gradients (either "fd" for finite-differences or "backprop" for backpropagation)

        Returns
        -------
        mse_train : numpy array
            mse on training data over training epochs
        mse_test : numpy array, optional
            mse on test data over training epochs
        """
        # initialize weights
        self.weights = []
        self.biases = []
        for i in range(len(self.hidden_layer_sizes)):
            if i == 0:
                dim1 = x.shape[1]
            else:
                dim1 = self.hidden_layer_sizes[i - 1]
            dim2 = self.hidden_layer_sizes[i]
            if (self.activations[i] == relu) or (self.activations[i] == identity) or (self.activations[i] == square):
                W = np.sqrt(2. / (dim1 + dim2)) * np.random.randn(dim2, dim1)
            else:
                raise ValueError("No initialization for layer with activation %s" % self.activations[i])
            b = np.zeros(dim2)

            self.weights.append(W)
            self.biases.append(b)

        # initialize optimizer
        self._optimizer = AdamOptimizer(
            self.weights + self.biases, self.lr, 0.9, 0.999,
            1e-08)

        # optimize weights and biases
        mse_train = []
        mse_test = []
        pbar = tqdm(range(n_epochs))
        for epoch in pbar:
            # show current MSE on training data
            pbar.set_description("MSE: %.2f" % self.mean_squared_error(x, y))

            # permute ordering of the samples
            _perm = np.random.permutation(x.shape[0])
            sidx = 0
            for batch in range(len(_perm) // bs + 1):
                eidx = sidx + bs
                if eidx > x.shape[0]:
                    eidx = x.shape[0]
                perm = _perm[sidx:eidx]

                # compute gradients
                gradients, gradients_biases = self.grad(x[perm], y[perm], method=method)

                # weight decay
                for w, g in zip(self.weights, gradients):
                    g += self.lr * self.alpha * 2 * w

                # update parameters
                self._optimizer.update_params(self.weights + self.biases, gradients + gradients_biases)

                # move to next batch of data
                sidx += bs
                if sidx >= x.shape[0]:
                    sidx = 0
                    break

            # record training/test performance for plotting
            mse_train.append(self.mean_squared_error(x_train, y_train))
            if x_test is not None:
                mse_test.append(self.mean_squared_error(x_test, y_test))

        if x_test is None:
            return mse_train
        else:
            return mse_train, mse_test

    def grad(self, x, y, method='backprop'):
        """
        Compute gradient (proxy method to call functions for computing gradients
        using either finite differences or backpropagation)
        """
        if method == 'backprop':
            return self.grad_backprop(x, y)
        elif method == 'fd':
            return self.grad_fd(x, y)
        else:
            raise ValueError("Invalid method for computing gradients (%s)." % method)

    def grad_fd(self, x, y, eps=1e-5):
        """Gradient computation by finite differences"""
        fg = []
        for W in self.weights:
            _W = np.copy(W)
            _fg = np.zeros_like(W)
            for i in range(W.shape[0]):
                for j in range(W.shape[1]):
                    W[i, j] = _W[i, j] + eps
                    f1 = self.mean_squared_error(x, y)
                    W[i, j] = _W[i, j] - eps
                    f2 = self.mean_squared_error(x, y)
                    W[i, j] = _W[i, j]
                    _fg[i, j] = (f1 - f2) / (2 * eps)
            fg.append(_fg)

        # finite difference gradients of biases
        fg2 = []
        for b in self.biases:
            _b = np.copy(b)
            _fg2 = np.zeros_like(b)
            for i in range(b.shape[0]):
                b[i] = _b[i] + eps
                f1 = self.mean_squared_error(x, y)
                b[i] = _b[i] - eps
                f2 = self.mean_squared_error(x, y)
                b[i] = _b[i]
                _fg2[i] = (f1 - f2) / (2 * eps)
            fg2.append(_fg2)

        return fg, fg2

    def grad_backprop(self, x, y):
        """
        This function performs the backpropagation algorithm inside the MLP class.
        It updates the weights inside the class.
        :param x: training design matrix
        :param y: training true target vector
        :return: [gradient of the weights, gradient of the biases] for each layer of the NN
        """
        # storage
        grad = []
        grad_bias = []
        delta = None

        # save the number of layers for readability
        num_layers = len(self.hidden_layer_sizes)

        # Compute the error done by the net, for the output layer use the MSE, for the internal backpropagate
        for i in np.flip(range(num_layers)):

            if i == num_layers - 1:
                # this is the output layer
                _, delta = self.mean_squared_error(x, y, cache=True, gradient=True)

            else:
                # those are the hidden layers
                delta = self.cache_derivatives[i].T * np.dot(delta, self.weights[i + 1]) delta @ self.weights[i + 1]

            # compute the gradient
            grad.append(np.dot(delta.T, self.cache_post_activations[i]))

            # append the bias
            grad_bias.append(np.array(delta.sum(axis=0)))

        # since I did all the computation backward, I need to revese the indices!
        grad = reversed(grad)
        grad_bias = reversed(grad_bias)

        # cast to list
        grad = list(grad)
        grad_bias = list(grad_bias)

        # clear the cache!
        self.cache_post_activations = []
        self.cache_derivatives = []

        print(grad[0].shape)

        # return grad, grad_biases
        return grad, grad_bias

    def mean_squared_error(self, x, y, cache=False, gradient=False):
        """Mean squared error loss function.

        Parameters
        ----------
        x : numpy array
            Input data (shape n_samples x n_features)
        y : numpy array
            Targets (shape n_samples x 1)
        cache : bool, optional
            If set to true, caches activations and derivatives of activation
            functions for usage during backprop.
        gradient : bool, optional
            A flag used to indicate whether the gradient of the activation should
            be computed.

        Returns
        -------
        err : float
            Squared loss.
        g : numpy array
            gradient of loss function
        """
        t = self.predict(x, cache=cache)
        err = np.mean((y - t) ** 2)
        if not gradient:
            return err
        else:
            g = -2 * (y - t)
            return err, g

    def predict(self, x, cache=False):
        """Compute predictions for a batch of samples.

        Parameters
        ----------
        x : numpy array
            Input data (shape n_samples x n_features)
        cache : bool, optional
            If set to true, caches activations and derivatives of activation
            functions for usage during backprop.

        Returns
        -------
        t : numpy array
            Predictions.
        """
        if cache:
            self.cache_post_activations.append(x.copy())
        t = x.T
        for i in range(len(self.hidden_layer_sizes)):
            t = np.matmul(self.weights[i], t) + np.expand_dims(self.biases[i], 1)
            if not cache:
                t = self.activations[i](t)
            else:
                t, g = self.activations[i](t, gradient=True)

            if cache:
                self.cache_post_activations.append(t.T)
                self.cache_derivatives.append(g.T)

        return t.T


if __name__ == "__main__":
    # Fix randonmness
    np.random.seed(1235)

    # Load data
    hf = h5py.File('regression.h5', 'r')
    x_train = np.array(hf.get('x_train'))
    y_train = np.array(hf.get('y_train'))
    x_test = np.array(hf.get('x_test'))
    y_test = np.array(hf.get('y_test'))
    hf.close()

    # normalize data
    scaler = preprocessing.StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # train MLP

    start = time.time()
    clf = MyMLP(hidden_layer_sizes=(10, 10,), activations=[relu, relu])
    mse_train, mse_test = clf.fit(x_train, y_train, x_test=x_test, y_test=y_test, n_epochs=500, method='backprop')
    end = time.time()

    # report training time
    print("Training took %s seconds." % (end - start))

    # plot performance curves
    plt.plot(mse_train, label='train')
    plt.plot(mse_test, label='test')
    plt.xlabel('epochs')
    plt.ylabel('mse')
    plt.legend()
    plt.savefig('learning-curves.pdf', bbox_inches='tight')
    plt.show()

    ## evaluate MLP
    y_pred = clf.predict(x_train)
    print("TRAIN ERROR SCIKIT:", metrics.mean_squared_error(y_pred, y_train))

    y_pred = clf.predict(x_test)
    print("TEST ERROR SCIKIT:", metrics.mean_squared_error(y_pred, y_test))
