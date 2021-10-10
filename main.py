# this is the main
import sys
import pandas as pd
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale


# custom
from Classes import Lin_custom
from Classes import Lin_GD
from Classes import DAO
from Classes import SubTask13
from Classes import SubTask21
from Classes import SubTask22

# random seed
random.seed(10567)

# TASK 1
# First subtask: version of the libs
print("Python version: ", sys.version)
print("Pandas version: ", pd.__version__)
print("Numpy version: ", np.__version__)

# paths
input_path = "input13.txt"
output_path = "output13.txt"
# I am setting my paths not to have problems with special characters
input_path = input_path.replace(os.sep, '/')  # not really necessary here
output_path = output_path.replace(os.sep, '/')

# initialize the DAO: see classes.py
dao = DAO(input_path,output_path, ".txt", ".txt")

sub13 = SubTask13(dao)
sub13.process()


# TASK 2
# subtask 1
# setting up the paths
input_path = "input21.csv"
output_path = "output21.csv"

# initialize the DAO: see classes.py
dao = DAO(input_path,output_path, ".csv", ".csv")
# initialize subtask22
speed_test = SubTask21(dao)
speed_test.process()


# subtask 2
input_path = "input22.csv"
output_path = "output22.csv"

# initialize the DAO: see classes.py
dao = DAO(input_path,output_path, ".csv", ".csv")
# initialize subtask22
stat_test = SubTask22(dao)
stat_test.process()
exit()


# Third task:
california = datasets.fetch_california_housing()
data = pd.DataFrame(california.data)
target = pd.Series(california.target)
X_train, X_test, y_train,  y_test = train_test_split(data, target, test_size=0.3)
print("The number of feature of this DF is: ", len(data.columns))
print("The number of instances in the train Df is: ", len(X_train))
print("The number of instances in the test Df is: ", len(X_test))

# I train my alg. with SK standard linear regressor
lin_reg = LinearRegression()
# fit
lin_reg.fit(X=X_train, y=y_train)
# prediction
predictions = lin_reg.predict(X=X_test)
print("The MSE of the lin. reg. is: ", mean_squared_error(y_true=y_test, y_pred=predictions))

# I train my alg. using my custom linear regressor
my_lin = Lin_custom()
# fit
my_lin.fit(X=X_train, y=y_train)
# prediction
predictions_custom = my_lin.predict(X=X_test)
print("The MSE of the CUSTOM lin. reg. is: ", mean_squared_error(y_true=y_test, y_pred=predictions_custom))

## Now i use a custom gradient descent linear regressor
GD_lin = Lin_GD()
# fit
GD_lin.fit(X_train, y_train)
# prediction
predictions_GD = GD_lin.predict(X_test)
print("The MSE of the GD lin. reg. is: ", mean_squared_error(y_true=y_test, y_pred=predictions_GD))
GD_lin.plot_mse()

# Now I preprocess my data by scaling them
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
y_train_scaled = minmax_scale(y_train)
X_test_scaled = scaler.fit_transform(X_test)
y_test_scaled = minmax_scale(y_test)
# I transform them into pandas DF ans series
X_train_scaled = pd.DataFrame(X_train_scaled)
y_train_scaled = pd.Series(y_train_scaled)
X_test_scaled = pd.DataFrame(X_test_scaled)
y_test_scaled = pd.Series(y_test_scaled)

# I repeat the previous data
GD_lin.fit(X_train_scaled, y_train_scaled)
# prediction
predictions_GD_scaled = GD_lin.predict(X_test_scaled)
print("The MSE of the GD lin. reg. is: ", mean_squared_error(y_true=y_test, y_pred=predictions_GD_scaled))
GD_lin.plot_mse()