# this is the main
import sys
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures

# custom
from Classes import Lin_custom
from Classes import Lin_GD
from Classes import DAO
from Classes import SubTask13
from Classes import SubTask21
from Classes import SubTask22

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
dao = DAO(input_path, output_path, ".txt", ".txt")

sub13 = SubTask13(dao)
sub13.process()

# TASK 2
# subtask 1
# setting up the paths
input_path = "input21.csv"
output_path = "output21.csv"

# initialize the DAO: see classes.py
dao = DAO(input_path, output_path, ".csv", ".csv")
# initialize subtask22
speed_test = SubTask21(dao)
speed_test.process()

# subtask 2
input_path = "input22.csv"
output_path = "output22.csv"

# initialize the DAO: see classes.py
dao = DAO(input_path, output_path, ".csv", ".csv")
# initialize subtask22
stat_test = SubTask22(dao)
stat_test.process()

# Third task:
california = datasets.fetch_california_housing()
data = pd.DataFrame(california.data, index=None)
target = pd.Series(california.target)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)  # :))
print("The number of feature of this DF is: ", len(data.columns))
print("The number of instances in the train Df is: ", len(X_train))
print("The number of instances in the test Df is: ", len(X_test))

# I train my alg. with SK standard linear regressor
lin_reg = LinearRegression(fit_intercept=False)
# fit
lin_reg.fit(X=X_train, y=y_train)
# prediction
predictions = lin_reg.predict(X=X_test)
predictions_train = lin_reg.predict(X=X_train)
print("The MSE (TRAIN) of the lin. reg. is: ", mean_squared_error(y_true=y_train, y_pred=predictions_train))
print("The MSE (TEST) of the lin. reg. is: ", mean_squared_error(y_true=y_test, y_pred=predictions))

# I train my alg. using my custom linear regressor
my_lin = Lin_custom()
# fit
my_lin.fit(X=X_train, y=y_train)
# prediction
predictions_custom = my_lin.predict(X=X_test)
predictions_custom_train = my_lin.predict(X=X_train)
print("The MSE (TEST) of the CUSTOM lin. is: ", mean_squared_error(y_true=y_test, y_pred=predictions_custom))
print("The MSE (TRAIN) of the CUSTOM lin. is: ", mean_squared_error(y_true=y_train, y_pred=predictions_custom_train))
# I print the w vector
print(my_lin.w)


# Now I use a custom gradient descent linear regressor
# scheduled learning rate
# TRAIN
GD_lin = Lin_GD()
# fit
GD_lin.fit(X_train, y_train,X_test, y_test, adaptative=False)
# prediction
predictions_GD = GD_lin.predict(X_test)
print("The non adaptive MSE of the GD lin. reg. is: ", mean_squared_error(y_true=y_test, y_pred=predictions_GD))
# MSE descent on the X_test DF
print("Custom GD printed")
GD_lin.plot_mse(adaptative=False)


# Bold driver adaptive learning rate
GD_lin = Lin_GD()
# fit
GD_lin.fit(X_train, y_train, X_test, y_test, adaptative=True)
# prediction
predictions_GD = GD_lin.predict(X_test)
print("The adaptive MSE of the GD lin. reg. is: ", mean_squared_error(y_true=y_test, y_pred=predictions_GD))
print("Custom GD printed")
print("Custom GD printed BOLD DRIVER")
GD_lin.plot_mse(adaptative=True)


# Now I preprocess my data by scaling them
scaler = MinMaxScaler()
# I scale all the data togheter
scaled_data = scaler.fit_transform(data)
# I use the same seed to have consistency
X_train_scaled, X_test_scaled = train_test_split(scaled_data, test_size=0.3, random_state=42)  # :))
X_train_scaled = pd.DataFrame(X_train_scaled, index=None)
X_test_scaled = pd.DataFrame(X_test_scaled, index=None)

# I perform the non adaptive GD on the newly scaled data
GD_lin.fit(X_train_scaled, y_train, X_test, y_test, adaptative=False)
# prediction
predictions_GD_scaled = GD_lin.predict(X_test_scaled)
print("The MSE of the GD lin. reg. is: ", mean_squared_error(y_true=y_test, y_pred=predictions_GD_scaled))
print("Custom GD min max scaled printed")
GD_lin.plot_mse(adaptative=False)

# I perform the adaptive GD on the newly scaled data
GD_lin.fit(X_train_scaled, y_train,X_test, y_test, adaptative=True)
# prediction
predictions_GD_scaled = GD_lin.predict(X_test_scaled)
print("The MSE of the GD lin. reg. is: ", mean_squared_error(y_true=y_test, y_pred=predictions_GD_scaled))
print("Custom GD min max scaled adaptative printed")
GD_lin.plot_mse(adaptative=True)

# Now I want to use a polynomial
degree = [2, 3, 4]
for k in degree:
    poly = PolynomialFeatures(k)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.fit_transform(X_test_scaled)
    # I train my alg. with SK standard linear regressor because it is fast :)
    lin_reg = LinearRegression(fit_intercept=False)
    # fit
    lin_reg.fit(X=X_train_poly, y=y_train)
    # prediction
    predictions = lin_reg.predict(X=X_test_poly)
    predictions_train = lin_reg.predict(X=X_train_poly)
    print("The MSE (TRAIN) of the lin. reg. with poly of order ", k, " is: ",
          mean_squared_error(y_true=y_train, y_pred=predictions_train))
    print("The MSE (TEST) of the lin. reg. with poly of order ", k, " is: ",
          mean_squared_error(y_true=y_test, y_pred=predictions))
