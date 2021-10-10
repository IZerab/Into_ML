import sys
import pandas as pd
import numpy as np
from os import path
from abc import ABC
from abc import abstractmethod
from sklearn.metrics import mean_squared_error
from numpy.linalg import norm
import matplotlib.pyplot as plt

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
        m = data.iloc[:, i:i+dimension].values.tolist()
        list_of_matrices.append(m)
    return list_of_matrices


