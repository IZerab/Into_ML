import sys
import pandas as pd
import numpy as np
from os import path
from abc import ABC
from abc import abstractmethod
from sklearn.metrics import mean_squared_error
from numpy.linalg import norm
import matplotlib.pyplot as plt
import math


def mat_mul(A, B):
    """
    Function that multiplies two matrices
    :param A: first matrix
    :param B: second matrix
    :return: the resulting matrix
    """
    AB = np.zeros((len(A), len(B)))
    # iter through the rows of the first matrix
    for i in range(len(A)):
        # iter through the columns of the second matrix
        for j in range(len(B[0])):
            # iter through the rows of the second matrix
            for k in range(len(B)):
                AB[i][j] += A[i][k] * B[k][j]
    return AB


def mat_reader(data, dimension):
    """
    This function extract n different matrices from data
    :param dimension: dimension of the matrices to extract
    :param data: data that store the matrices
    :return: a list containing all the matrices extracted
    """
    list_of_matrices = []
    for i in range(0, len(data.columns), dimension):
        m = data.iloc[:, i:i + dimension].values.tolist()
        list_of_matrices.append(m)
    return list_of_matrices


def mean(array):
    """
    It calculates the mean of the array
    :param array: list or pd.Series
    :return: mean of the array
    """
    n = len(array)
    get_sum = sum(array)
    return get_sum / n


def median(array):
    """
    It calculates the median of the array
    :param array: list or pd.Series
    :return: median of the array
    """
    # list of elements to calculate median
    n = len(array)
    array.sort()
    if n % 2 == 0:
        median_down = array[n // 2 - 1]
        median_up = array[n // 2]
        median = (median_up + median_down) / 2
    else:
        median = array[n // 2]
    return median


def std(array):
    """
    It calculates the median of the array
    :param array: list or pd.Series
    :return: std of the array
    """
    n = len(array)
    # mu is the mean
    mu = mean(array)
    # square dev
    diff = [(x - mu) ** 2 for x in array]
    variance = sum(diff) / n
    std = math.sqrt(variance)
    return std


def quantiles(array):
    """
    It calculates the 0.25, 0.5 and 0.75 quantiles of the array
    :param array: list or pd.Series
    :return: list with the quantiles
    """
    array.sort()
    # I cast to int
    m = int(len(array)/2)
    # 0.25 quantile
    Q_25 = median(array[:m])
    # 0.5 quantile (median)
    Q_5 = median(array)
    # 0.75 quantile
    Q_75 = median(array[m:])
    return [Q_25, Q_5, Q_75]


def cus_min(array):
    """
    It calculate the smallest value of the array
    :param array: list or pd.Series
    :return:
    """
    array.sort()
    return array[0]


def cus_max(array):
    """
        It calculate the smallest value of the array
        :param array: list or pd.Series
        :return:
        """
    array.sort()
    array = array[::-1]
    return array[0]




