import pandas as pd
import numpy as np
import timeit
from os import path
from abc import ABC
from abc import abstractmethod
from sklearn.metrics import mean_squared_error
from numpy.linalg import norm
import matplotlib.pyplot as plt
from Functions import mat_mul
from Functions import mat_reader

# DAO
class DAO:
    """
    This class load the data from a memory inside the computer following the path.
    It can also save them in another location in the computer.
    When initializing this function you should provide path in and path out.

    """

    def __init__(self, ifname, ofname, iftype, oftype):
        """
        Initializing function
        :param ifname: path in
        :param ofname: path out
        :param iftype: type of input data (must be either .csv or .txt)
        :param oftype: type of output data (must be either .csv or .txt)
        """
        inputs = [ifname, ofname, iftype, oftype]
        files = [ifname, ofname]
        types = [iftype, oftype]
        for i in inputs:
            if not isinstance(i, str):
                raise ValueError("Inputs are not strings. Please amend that!")

        if not iftype == oftype:
            raise Exception("The input and output files are not of the same type!")



        if path.exists(ifname):
                self.ifname = ifname
        else:
            raise Exception("Input path do not exist. Please insert the paths again")
        self.ofname = ofname


        for i in range(len(types)):
            if types[i] == ".csv":
                if i == 0:
                    self.iftype = iftype
                if i == 1:
                    self.oftype = oftype
            elif types[i] == ".txt":
                if i == 0:
                    self.iftype = iftype
                if i == 1:
                    self.oftype = oftype
            else:
                raise Exception("The type of file to read is not supported!")

        # check if the passed types are .csv or .txt
        for i in range(len(types)):
            if types[i] == ".csv":
                # check if the passed filenames corresponds to the if/of types
                if not files[i].endswith(".csv"):
                    raise ValueError("The file and the type of file do not correspond!")
            elif types[i] == ".txt":
                # check if the passed filenames corresponds to the if/of types
                if not files[i].endswith(".txt"):
                    raise ValueError("The file and the type of file do not correspond!")

    def load(self):
        """
        Function that loads the data and store them inside a pd.DataFrame
        :return:
        """
        if self.iftype == ".csv":
            self.data = pd.read_csv(self.ifname, header=None)
        elif self.iftype == ".txt":
            self.data = pd.read_csv(self.ifname, sep=" ", header=None)
        return self.data

    def store(self, data_to_store):
        """
        Function that gets some data and store them in a csv or txt file.
        :param data_to_store: data to store
        :return: a boolean value to confirm the operation
        """
        print(data_to_store)
        if isinstance(data_to_store, list):
            data_to_store = np.concatenate(data_to_store)
        if isinstance(data_to_store, str):
            data_to_store = [data_to_store]
        data_to_store = pd.DataFrame(data_to_store)
        print(data_to_store)
        if self.oftype == ".csv":
            data_to_store.to_csv(self.ofname, header=None, index=None)
            return True
        if self.oftype == ".txt":
            data_to_store.to_csv(self.ofname, sep=" ", header=None, index=None)
            return True
        return False

# Abstract class
class SubTaskABC(ABC):
    @abstractmethod
    def process(self):
        pass

# Now I create the concrete class inheriting from the abstract class
class SubTask13(SubTaskABC):
    """
    Class with a DAO included. It inherits from the abstract subclass SubTaskABC.
    """
    def __init__(self, dao):
        """
        :param dao: DAO object that must be initialised before passing
        """
        self.my_dao = dao

    def process(self):
        """
        Function that stores the data inside a local variable, it sort all even indexed chars in increasing
        and odd indexed chars in decreasing order and finally save them in the memory disk.
        """
        data = self.my_dao.load()
        data = data.to_numpy().item()
        data = list(data)
        even = []
        odd = []
        for i in range(len(data)):
            if ((i % 2) == 0):
                even.append(data[i])
            else:
                odd.append(data[i])
        even = even[::-1]
        result = ''.join(odd+even)
        if not self.my_dao.store(result):
            raise Exception("The data were not stored correctly!")



class SubTask21(SubTaskABC):
    """
    It inherits from the abstract subclass SubTaskABC. In the initialization it requires a DAO to be passed as
    an argument.
    """

    def __init__(self, dao):
        self.my_dao = dao
        self.repetitions = 100000

    def process(self):
        """
        Function that stores the matrices inside a local variable using the DAO, it multiplies the first two matrices
        by using a custom matrix multiplication function and using numpy. It also measures the execution time of both
        the techniques as an average over 10^5 iterations. Finally it save the result of the multiplication in the
        memory disk where it also reports the runtime.
        """
        # I read the matrices (the file is in the same folder)
        # I import the matrices as a whole
        matrices = self.my_dao.load()
        matrices = mat_reader(matrices, 3)
        print(type(matrices))
        print(matrices)
        # I set the runtime to 0 to take the temporal intervals
        sum_runtime = 0
        for k in range(self.repetitions):
            A = matrices[0]
            for i in range(len(matrices) - 1):
                B = matrices[i + 1]
                time_1 = timeit.default_timer()
                C = mat_mul(A, B)
                A = C
                # the time elapsed is calculated through a difference
                time_2 = timeit.default_timer() - time_1
                sum_runtime += time_2
        # I average over 10^5 iterations
        avg_custom = sum_runtime / self.repetitions

        sum_runtime = 0
        # I perform the same with the numpy native function
        for k in range(self.repetitions):
            A = matrices[0]
            for i in range(len(matrices) - 1):
                B = matrices[i + 1]
                time_1 = timeit.default_timer()
                C = np.dot(A, B)
                A = C
                # the time elapsed is calculated through a difference
                time_2 = timeit.default_timer() - time_1
                sum_runtime += time_2

        # I average over 10^5 iterations
        avg_numpy = sum_runtime / self.repetitions
        print("Numpy: ", avg_numpy)
        print("Custom: ", avg_custom)
        speedup = avg_custom/avg_numpy
        print("The speedup obtained through the use of custom function of  is: ", speedup)
        # storing the results
        self.my_dao.store([C])
        with open('output21.csv', 'a') as file:
            np.savetxt(file, [speedup], fmt='%.3f')

class Lin_custom():
    """
    This class is a custom lin regression class.
    The weight vector is found with the LSLR close form.
    It includes the fit and the predict method.
    """

    def __init__(self):
        # parameter vector for the linear regression
        self.w = pd.Series()

    def fit(self, X, y):
        """
        Manually implementing the linear fit. We use the lin regressor close form!
        :param X: X input matrix as pandas DF
        :param y: y target vector as pandas Series
        :return: nothing, it saves W in the class
        """
        if isinstance(X, pd.DataFrame):
            X = X
        else:
            raise ValueError("Value error: The X input is not a pandas DF!")
        if isinstance(y, pd.Series):
            y = y
        else:
            raise ValueError("Value error: The y input is not a pandas Series!")

        # I traspone X
        Xt = X.transpose()
        # I multiply Xt and X
        temp = np.dot(Xt, X)
        # I compute the inverted matrix of temp, NOTE that temp is a square matrix
        inverse = np.linalg.inv(temp)
        # I multiply for Xt
        temp2 = np.dot(inverse, Xt)
        # I multiply for y
        # This is the parameter we were looking for!! It is saved
        self.w = np.dot(temp2, y)

    def predict(self, X):
        """
        This function predicts, based on the previous fit, what the y_predicted will be.
        :param X:
        :return: y_predicted, the predicted values in our learning problem
        """
        y_pred = np.dot(X, self.w)
        return y_pred


class Lin_GD():
    """
    This class is a custom lin regression class.
    The weight vector is found with the gradient descent.
    It includes the fit and the predict method.
    """

    def __init__(self):
        # parameter vector for the linear regression
        self.w = pd.Series()

    def fit(self, X, y):
        """
        Manually implementing the linear fit. We use the gradient descent to find it!
        The cost function of my choice is the MSE.
        :param X: X input matrix as pandas DF
        :param y: y target vector as pandas Series
        :return: nothing, it saves W in the class
        """
        if isinstance(X, pd.DataFrame):
            X = X
        else:
            raise ValueError("Value error: The X input is not a pandas DF!")
        if isinstance(y, pd.Series):
            y = y
        else:
            raise ValueError("Value error: The y input is not a pandas Series!")

        # I set w to be a zero vector
        w = np.zeros(len(X.columns))
        # I transform it into a pd.Series
        w = pd.Series(w)
        # the cost function that I choose is the MSE, I store the results to create a plot
        mse = []
        # Learning rate of my choice
        l_rate = 0.0000001
        # I set a precision value to stop the algorithm
        precision = 0.0000000001  # this is not necessary since we instantly get the minimum
        # for convenience I set the variable diff_w (lenght of the step) as 1
        len_step = 0.1
        # I also put a stopping condition iter > 1000000
        max_iter = 100005
        counter = 0

        while len_step > precision and counter < max_iter:  # since we instantly get the minimum I do not include "len_step > precision and"
            # I write the gradient of the MSE for a linear regression, I used matrix formalism
            distance = np.dot(X, w) - y
            distance = pd.Series(distance)
            temp2 = distance.transpose()
            temp3 = np.dot(temp2, X)
            gradient = 2 / len(X) * temp3
            # I use gradient descent
            w_new = w - l_rate * gradient
            # I save the MSE in the mse list
            len_step = norm(w_new - w)
            y_pred = np.dot(X, w)
            mse.append(mean_squared_error(y_pred, y))
            # I update the weight vector
            w = w_new
            counter = counter + 1

        # I save my result in my class
        self.w = w
        self.mse = mse

    def plot_mse(self):
        """
        Function that plots mse against the number of iteration.
        :return:
        """
        # first I select only the required parameters (1+1000k where k belongs to (1:100))
        supp = []
        for k in range(100):
            supp.append(self.mse[1 + 1000 * k])
        x_plot = [1 + 1000 * k for k in range(100)]
        plt.scatter(x_plot, supp)
        plt.ylabel('MSE')
        plt.xlabel('Number of iteration')

    def predict(self, X):
        """
        This function predicts, based on the previous fit, what the y_predicted will be.
        :param X:
        :return: y_predicted, the predicted values in our learning problem
        """
        y_pred = np.dot(X, self.w)
        return y_pred
