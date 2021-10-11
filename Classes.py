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
from Functions import cus_mean
from Functions import cus_median
from Functions import quantiles
from Functions import cus_max
from Functions import cus_min
from Functions import std


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
        if isinstance(data_to_store, dict):
            # I transpose for a better readability
            data_to_store = pd.DataFrame.from_dict(data_to_store, orient='index')
            if self.oftype == ".csv":
                data_to_store.to_csv(self.ofname, header=None, float_format="%.2f")
                return True
            if self.oftype == ".txt":
                data_to_store.to_csv(self.ofname, sep=" ", header=None, float_format="%.2f")
                return True
        if isinstance(data_to_store, list):
            data_to_store = np.concatenate(data_to_store)
        if isinstance(data_to_store, str):
            data_to_store = [data_to_store]
        data_to_store = pd.DataFrame(data_to_store, index=None)
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

        # CUSTOM
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

        # NUMPY
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
        speedup = avg_custom / avg_numpy
        print("The speedup obtained through the use of custom function of  is: ", speedup)
        # storing the results
        self.my_dao.store([C])
        with open('output21.csv', 'a') as file:
            np.savetxt(file, [speedup], fmt='%.3f')


class SubTask22(SubTaskABC):
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
        matrix = self.my_dao.load()
        # dictionary where to have the results
        results = {
            "mean_pd": [],
            "median_pd": [],
            "std_pd": [],
            "25_pd": [],
            "5_pd": [],
            "75_pd": [],
            "min_pd": [],
            "max_pd": [],
            "mean_cus": [],
            "median_cus": [],
            "std_cus": [],
            "25_cus": [],
            "5_cus": [],
            "75_cus": [],
            "min_cus": [],
            "max_cus": [],
        }

        # I separate the two columns
        columns = [matrix.values[:, 0], matrix.values[:, 1]]
        # PANDAS
        # I set the runtime to 0 to take the temporal intervals
        sum_runtime = 0
        # dictionary where to have the results
        for k in range(len(columns)):
            temp = pd.Series(columns[k])
            time_1 = timeit.default_timer()
            results["mean_pd"].append(temp.mean())
            results["median_pd"].append(temp.median())
            results["std_pd"].append(temp.std())
            # the quantile methods give an array as the output
            temp2 = temp.quantile([.25, .5, .75]).values
            results["25_pd"].append(temp2[0])
            results["5_pd"].append(temp2[1])
            results["75_pd"].append(temp2[2])
            results["min_pd"].append(temp.min())
            results["max_pd"].append(temp.max())
            time_2 = timeit.default_timer() - time_1
            sum_runtime += time_2
        # I average over 10^5 iterations
        avg_pd_sep = sum_runtime / self.repetitions

        # CUSTOM
        sum_runtime = 0
        # I perform the same with the numpy native function
        for k in range(len(columns)):
            time_1 = timeit.default_timer()
            results["mean_cus"].append(cus_mean(columns[k]))
            results["median_cus"].append(cus_median(columns[k]))
            results["std_cus"].append(columns[k].std())
            # the quantile methods give an array as the output
            temp2 = quantiles(columns[k])
            results["25_cus"].append(temp2[0])
            results["5_cus"].append(temp2[1])
            results["75_cus"].append(temp2[2])
            results["min_cus"].append(min(columns[k]))
            results["max_cus"].append(max(columns[k]))
            time_2 = timeit.default_timer() - time_1
            sum_runtime += time_2
        # I average over 10^5 iterations
        avg_cus_sep = sum_runtime / self.repetitions

        # I compute the statistics for the two columns togheter
        # I create a single long list with the two matrices
        matrix = [*matrix.values[:, 0], *matrix.values[:, 1]]

        # CUSTOM
        # I set the runtime to 0 to take the temporal intervals
        sum_runtime = 0
        # Operations
        time_1 = timeit.default_timer()
        results["mean_cus"].append(cus_mean(matrix))
        results["median_cus"].append(cus_median(matrix))
        results["std_cus"].append(std(matrix))
        # matrix output: index is q, the columns are the columns of matrix, and the values are the quantiles.
        temp2 = quantiles(matrix)
        results["25_cus"].append(temp2[0])
        results["5_cus"].append(temp2[1])
        results["75_cus"].append(temp2[2])
        results["min_cus"].append(cus_min(matrix))
        results["max_cus"].append(cus_max(matrix))
        time_2 = timeit.default_timer() - time_1
        sum_runtime += time_2
        # I average over 10^5 iterations
        avg_together_cus = sum_runtime / self.repetitions

        # PANDAS
        # I set the runtime to 0 to take the temporal intervals
        matrix = pd.Series(matrix)
        sum_runtime = 0
        # Operations
        time_1 = timeit.default_timer()
        results["mean_pd"].append(matrix.mean())
        results["median_pd"].append(matrix.median())
        results["std_pd"].append(matrix.std())
        # matrix output: index is q, the columns are the columns of matrix, and the values are the quantiles.
        temp2 = matrix.quantile([0.25, 0.5, 0.75]).values
        results["25_pd"].append(temp2[0])
        results["5_pd"].append(temp2[1])
        results["75_pd"].append(temp2[2])
        results["min_pd"].append(matrix.min())
        results["max_pd"].append(matrix.max())
        time_2 = timeit.default_timer() - time_1
        sum_runtime += time_2
        # I average over 10^5 iterations
        avg_together_pd = sum_runtime / self.repetitions

        speedups = [avg_together_cus / avg_together_pd, avg_cus_sep / avg_pd_sep]
        # storing the results
        self.my_dao.store(results)
        with open('output22.csv', 'a') as file:
            np.savetxt(file, speedups, fmt='%.3f')


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
        result = ''.join(odd + even)
        if not self.my_dao.store(result):
            raise Exception("The data were not stored correctly!")


class Lin_custom():
    """
    This class is a custom lin regression class.
    The weight vector is found with the LSLR close form.
    It includes the fit and the predict method.
    """

    def __init__(self):
        # parameter vector for the linear regression
        self.w = None

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
        # This is the parameter we were looking for!! It is saved inside the class
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
        self.w = []
        self.mse = []
        self.mse_test = []

    def fit(self, X, y, X_test, y_test, adaptative):
        """
        Manually implementing the linear fit. We use the gradient descent to find it!
        The cost function of my choice is the MSE.
        :param y_test: test target, it is needed to plot the mse descent on test
        :param adaptative: control parameter, Boolean, to check if to use the updated learning rate
        :param X_test: test dataframe, it is needed to plot the mse descent on test
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
        mse_test = []
        # Learning rate of my choice
        l_rate = 0.0000001
        # I set a precision value to stop the algorithm
        precision = 0.0000000001  # this is not necessary since we instantly get the minimum
        # for convenience I set the variable diff_w (lenght of the step) as 1
        len_step = 0.1
        # I also put a stopping condition iter > 5005
        max_iter = 5005
        counter = 0

        while len_step > precision and counter < max_iter:
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
            y_pred_test = np.dot(X_test, w)

            mse_test.append(mean_squared_error(y_pred_test, y_test))
            mse.append(mean_squared_error(y_pred, y))
            # I update the weight vector
            w = w_new
            if not adaptative:
                # this is a REALLY ROUGH condition, I first plotted the graph of mse vs iterations and got where the
                # elbow of the function was and later I adjusted the learning rate for the iterations after it.
                #  better solution is presented in subtask 3.5
                if counter == 2000:
                    l_rate = l_rate/counter*10
            if adaptative:
                # I choose to use the bold driver heuristic
                C_acc = 1.05
                C_dec = 0.8
                # I avoid to adjust the index at the first iterations for convenience
                if counter > 2:
                    if mse[counter] > mse[counter-1]:
                        l_rate = l_rate*C_dec
                    else:
                        l_rate = l_rate * C_acc
            counter = counter + 1
        # I save my result in my class
        self.w = w
        self.mse = mse
        self.mse_test = mse_test

    def plot_mse(self, adaptative):
        """
        Function that plots mse against the number of iteration.
        :return: the plot
        """
        # first I select only the required parameters (1+1000k where k belongs to (1:100) was too much!)
        supp = []
        # FOR THE TRAIN
        if not adaptative:
            for k in range(20):
                supp.append(self.mse[1 + 250 * k])
            x_plot = [1 + 250 * k for k in range(20)]
            plt.plot(x_plot, supp,'bo', linestyle='dashed')
            plt.ylabel('MSE train')
            plt.xlabel('Number of iterations')
            plt.show()
        if adaptative:
            for k in range(25):
                supp.append(self.mse[1 + 45 * k])
            x_plot = [1 + 45 * k for k in range(25)]
            plt.plot(x_plot, supp,'bo', linestyle='dashed')
            plt.ylabel('MSE train')
            plt.xlabel('Number of iterations')
            plt.show()

        supp = []
        # FOR THE TEST
        supp = []
        if not adaptative:
            for k in range(20):
                supp.append(self.mse_test[1 + 250 * k])
            x_plot = [1 + 250 * k for k in range(20)]
            plt.plot(x_plot, supp,'bo', linestyle='dashed')
            plt.ylabel('MSE test')
            plt.xlabel('Number of iterations')
            plt.show()
        supp = []
        if adaptative:
            for k in range(20):
                supp.append(self.mse_test[1 + 250 * k])
            x_plot = [1 + 250 * k for k in range(20)]
            plt.plot(x_plot, supp, 'bo', linestyle='dashed')
            plt.ylabel('MSE test')
            plt.xlabel('Number of iterations')
            plt.show()

    def predict(self, X):
        """
        This function predicts, based on the previous fit, what the y_predicted will be.
        :param X:
        :return: y_predicted, the predicted values in our learning problem
        """
        y_pred = np.dot(X, self.w)
        return y_pred
