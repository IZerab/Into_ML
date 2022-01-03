import h5py
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from stochastic_optimizers import AdamOptimizer
import matplotlib.pyplot as plt
import time


class MLP_custom(object):
    """
    Complete class for custom MLP
    together with all necessary methods for Task 2
    """
    def __init__(self, layers):
        self.W = []
        self.B = []
        self.optimizer = None

        # initialize weights
        for i in range(len(layers) - 1):
            w = np.random.normal(loc=0.0, scale=(2 / layers[i+1]), size=(layers[i], layers[i+1]))
            self.W.append(w)

        # initialize biases
        for i in range(1, len(layers) - 1):
            self.B.append(np.zeros((1, layers[i])))

    def get_params(self):
        return self.W, self.B

    def set_optimizer(self, lr, eps):
        optimizer = AdamOptimizer(self.W + self.B, lr, 0.9, 0.999, eps)
        self.optimizer = optimizer

    def forward(self, X, W_, B_):
        # initialize v
        v = X
        # perform forward step
        for i in range(len(W_) - 1):
            z = np.dot(v, W_[i]) + B_[i]
            v = self.relu(z)

        # return output (no activation and bias here)
        return np.dot(v, W_[-1])

    def relu(self, X):
        # compute relu for every element of X
        return np.maximum(0, X)

    def cost(self, pred, y):
        # compute SE for the predicted output
        cost = np.sum((pred - y) ** 2)
        return cost

    def mse(self, pred, y):
        cost = np.sum((pred - y) ** 2) / len(y)
        return cost

    def finite_difference(self, pred1, pred2, y, e):
        l1 = self.cost(pred1, y)
        l2 = self.cost(pred2, y)
        return (l1 - l2) / 2 * e

    # computes the partial derivative of the loss function
    # with respect to the weights and the bias terms
    def compute_gradients(self, X, y, ws, bs, epsilon):
        # list of weight gradient
        dW = []
        dB = []

        # iterate though all weight matrices
        for k in range(len(ws)):
            w = ws[k]
            dw = np.zeros(w.shape)
            # compute finite difference for every element
            for i in range(w.shape[0]):
                for j in range(w.shape[1]):
                    # add epsilon to one component
                    w[i, j] += epsilon
                    pred1 = model.forward(X, ws, bs)
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

    def train(self, x_train, y_train, x_test, y_test, max_iter=10000, epsilon=0.001):
        # initialize necessary variables
        train_mses = []
        test_mses = []
        iterations = []
        old_mse = 0

        print('Training started...')
        for i in range(1, max_iter):
            # compute gradients
            gradient_w, gradient_b = self.compute_gradients(x_train, y_train, self.W, self.B, epsilon)
            # update parameters
            self.optimizer.update_params(self.W + self.B, gradient_w + gradient_b)

            # compute loss
            if i % 100 == 0:
                # evaluate on training set
                pred = self.forward(x_train, self.W, self.B)
                mse = self.mse(pred, y_train)
                print('Iteration:', i, 'MSE:', mse)

                # the testing set is used ONLY for evaluation here
                pred_test = self.forward(x_test, self.W, self.B)
                mse_test = self.mse(pred_test, y_test)

                # check for convergence (stop if there is little improvement)
                if np.abs(old_mse - mse) < 0.05:
                    print('Converged.')
                    break

                # store current results
                old_mse = mse
                train_mses.append(mse)
                test_mses.append(mse_test)
                iterations.append(i)

        print('Final testing MSE:', test_mses[-1])
        return train_mses, test_mses, iterations


"""
########  SUBTASK 2.1  ########
"""

# read the data
hf = h5py.File("regression.h5", 'r')
x_train = np.array(hf.get("x_train"))
y_train = np.array(hf.get("y_train"))
x_test = np.array(hf.get("x_test"))
y_test = np.array(hf.get("y_test"))
hf.close()

# scale data with min-max scaler
sc = MinMaxScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

# reporting statistics
print('Number of features:', x_train.shape[1])
print('Training sample size:', x_train.shape[0])
print('Testing sample size:', x_test.shape[0])


"""
########  SUBTASK 2.2 ########
"""

# define layer sizes
layer_sizes = [x_train.shape[1], 10, 10, 1]
# initialize MLP model defined above
model = MLP_custom(layer_sizes)

# test the forward propagation method:
W0, B0 = model.get_params()
y_hat = model.forward(x_train, W0, B0)
print('Forward propagation output:', y_hat.shape)


"""
########  SUBTASK 2.3  ########
"""

# define parameters for Adam
lr = 0.05
epsilon = 1e-06
# set the optimizer
model.set_optimizer(lr, epsilon)

# start the training
start_time = time.time()
train_mses, test_mses, iterations = model.train(x_train, y_train, x_test, y_test)
end_time = time.time()
print("Execution time for training is", end_time - start_time, "seconds.")

# plot MSEs
plt.plot(iterations, train_mses, label='Training set')
plt.plot(iterations, test_mses, label='Testing set')
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.legend()
plt.show()