#----------------------------------------------------------------
#
# Lukas Till Schawerda
# Introduction to Machine Learning (VU)
# Programming assignment 3 / due 29.11.2021
#
#----------------------------------------------------------------


#----------------------------------------------------------------
#
# Task 1: Feature selection
#
#----------------------------------------------------------------

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, preprocessing, model_selection, svm, metrics

# Breast cancer data set
data_raw1 = datasets.load_breast_cancer()
names1 = data_raw1.feature_names

# Extract x, y
x_raw1 = data_raw1.data
y_raw1 = data_raw1.target

# Apply MinMaxScaler
x_mms = preprocessing.MinMaxScaler().fit_transform(x_raw1)

# Random state
seed = 2021
np.random.seed(seed)

# Shuffle the data beforehand
ind_shuffle = np.random.permutation(x_mms.shape[0])
x_shuffle = x_mms[ind_shuffle]
y_shuffle = y_raw1[ind_shuffle]

# Split data set
x_train1, x_test1, y_train1, y_test1 = model_selection.train_test_split(
    x_shuffle, y_shuffle, test_size=0.3, random_state=seed)


#----------------------------------------------------------------
# Subtask 1.1: Forward greedy feature selection
#----------------------------------------------------------------

# Function for greedy forward selection

def greedy_forward_selection(x, y, k=10, num_features=None):

    # Number of potential features
    p = x.shape[1]
    
    # Initialize elements
    potential_features = np.arange(p)
    selected_features = []
    performance = {}

    # Iterate until max number of features
    while len(selected_features) < p:
        
        # List to store accuracies
        accuracies = np.zeros(p)
        
        # For each feature that might be added
        for feat in potential_features:
            
            # Features currently being modeled
            model_features = selected_features + [feat]
            
            # Linear SVM model
            mod = svm.LinearSVC()
    
            # Compute accuracy
            acc = model_selection.cross_val_score(mod, x[:, model_features], y,  cv = k)
            
            # Store accuracy
            accuracies[feat] = acc.mean()

        # Best accuracy
        ind_best = np.argmax(accuracies)
        
        # Add new feature
        selected_features.append(ind_best.copy())
            
        # Remove feature from potential feature list
        potential_features = list(set(potential_features) - set(selected_features))
            
        # Record performance
        performance[len(selected_features)] = accuracies[ind_best]
        
        # If number of features is specified and reached, break
        if num_features is not None and len(selected_features) == num_features:
            break
    
    # Return results
    return selected_features, performance
        

# Compute forward greedy feature selection
features_forward, performance_forward = greedy_forward_selection(x_train1, y_train1)


# Report order and names
for i in features_forward:
    if i==features_forward[0]: print("Using greedy forwards selection, the features are added in this order:")
    print("(" + str(i)+") " + names1[i])

# Report performance and number of features
for j in performance_forward:
    if j==1: print("Using greedy forwards selection, the performance (accuracy) of the different sized models is")
    print(j, "features:", np.round(performance_forward[j], 6))

# Plotting the performance
fig, ax = plt.subplots()
xx1 = [p for p in performance_forward]
yy1 = [performance_forward[p] for p in performance_forward]
ax.plot(xx1, yy1)
plt.title("Forward greedy feature selection CV performance")
plt.xlabel("Number of features")
plt.ylabel("Accuracy")
plt.show()


#----------------------------------------------------------------
# Subtask 1.2: Backward greedy feature selection
#----------------------------------------------------------------

# Function for greedy backward selection

def greedy_backward_selection(x, y, k=10, num_features=None):

    # Number of potential features
    p = x.shape[1]

    # Initialize elements
    selected_features = np.arange(p)
    removed_features = []
    performance = {}
    
    # Linear SVM model
    mod = svm.LinearSVC()
    
    # Accuracy of full model
    acc = model_selection.cross_val_score(mod, x, y, cv = k)
    
    acc_full = acc.mean()
    performance[p] = acc_full
    
    # Iterate until all features are removed
    while len(selected_features) > 1:
        
        # List to store accuracies
        accuracies = np.zeros(p)
        
        # For each feature that might be added
        for feat in selected_features:
            
            # Features currently being modeled
            model_features = list(set(selected_features) - set([feat]))
            
            # Linear SVM model
            mod = svm.LinearSVC()
            
            # Compute accuracy
            acc = model_selection.cross_val_score(mod, x[:, model_features], y, cv = k)
            
            # Store accuracy
            accuracies[feat] = acc.mean()

        # Best accuracy, meaning which feature can we
        # leave out to still obtain the best accuracy
        ind_best = np.argmax(accuracies)
        
        # Add feature to list of removed features
        removed_features.append(ind_best.copy())
            
        # Remove feature from list of selected features
        selected_features = list(set(selected_features) - set([ind_best]))
        
        # Record performance
        performance[len(selected_features)] = accuracies[ind_best]
        
        # If number of features is specified and reached, break
        if num_features is not None and len(selected_features) == num_features:
            break
    
    # Return results
    return selected_features, removed_features, performance

# Compute backward greedy feature selection
features_backward, removed_backward, performance_backward = greedy_backward_selection(x_train1, y_train1)


# Report order and names of removed features
for i in removed_backward:
    if i==removed_backward[0]: print("Using greedy backwards selection, the features are removed in this order:")
    print("(" + str(i)+") " + names1[i])

# Report final model
print("The last feature left is ("+str(features_backward[-1])+") "+names1[features_backward[-1]])


# Report performance and number of features
for j in performance_backward:
    if j==30: print("Using greedy backwards selection, the performance (accuracy) of the different sized models is")
    print(j, "features:", np.round(performance_backward[j], 6))


# Plotting the performance
fig, ax = plt.subplots()
xx2 = [p for p in performance_backward]
yy2 = [performance_backward[p] for p in performance_backward]
ax.plot(xx2, yy2)
plt.title("Backward greedy feature selection CV performance")
plt.xlabel("Number of features")
plt.ylabel("Accuracy")
plt.show()



#----------------------------------------------------------------
# Subtask 1.3: Feature importance
#----------------------------------------------------------------

# --- Comparing forward and backward ---

fig, ax = plt.subplots()
ax.plot(xx1, yy1, label="Forward")
ax.plot(xx2, yy2, label="Backward")
plt.vlines(6, np.min(yy1+yy2), np.max(yy1+yy2), label="6 features")
plt.title("Greedy feature selection CV performance")
plt.xlabel("Number of features")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# ---

# Specify number of features
num_features = 6

# Forward greedy feature selection with 6 features
features_forward6, performance_forward6 = greedy_forward_selection(x_train1, y_train1, num_features=num_features)

# Backward greedy feature selection with 6 features
features_backward6,removed_backward6,  performance_backward6 = greedy_backward_selection(x_train1, y_train1, num_features=num_features)


# Report forward final model
for i in features_forward6:
    if i==features_forward6[0]: print("Using greedy forwards selection, the final model consists of these 6 features:")
    print("(" + str(i)+") " + names1[i])

# Report backward final model
for m in features_backward6:
    if m==features_backward6[0]: print("Using greedy backwards selection, the final model consists of these 6 features:")
    print("(" + str(m)+") " + names1[m])


# Compare CV training performance
print("Forward CV training accuracy:", np.round(performance_forward6[6],6))
print("Backward CV training accuracy:", np.round(performance_backward6[6],6))


# Compare performance on test data
best_forward6 = svm.LinearSVC()
best_backward6 = svm.LinearSVC()

# Best models
best_forward6.fit(x_train1[:, features_forward6], y_train1)
best_backward6.fit(x_train1[:, features_backward6], y_train1)

# Prediction
pred_forward6 = best_forward6.predict(x_test1[:, features_forward6])
pred_backward6 = best_backward6.predict(x_test1[:, features_backward6])

# Accuracy
acc_forward6 = metrics.accuracy_score(y_test1, pred_forward6)
acc_backward6 = metrics.accuracy_score(y_test1, pred_backward6)

# Compare test performance
print("Forward test accuracy:", np.round(acc_forward6,6))
print("Backward test accuracy:", np.round(acc_backward6,6))


#----------------------------------------------------------------
#
# Task 2: Kernelized SVM
#
#----------------------------------------------------------------

# Read training & test data
data_raw_train2 = datasets.fetch_20newsgroups_vectorized(subset="train")
data_raw_test2 = datasets.fetch_20newsgroups_vectorized(subset="test")
names2 = data_raw_train2.target_names

x_raw_train2 = data_raw_train2.data
y_raw_train2 = data_raw_train2.target

x_raw_test2 = data_raw_test2.data
y_raw_test2 = data_raw_test2.target


# Take subset to help computation
n_train = 800
n_test = int(0.5 * n_train)

# Random state
seed = 12
np.random.seed(seed)

# Shuffle the data beforehand
ind_train = np.random.permutation(x_raw_train2.shape[0])
ind_test = np.random.permutation(x_raw_test2.shape[0])

# Take subset
ind_train = ind_train[:n_train]
ind_test = ind_test[:n_test]

# Training data
x_train2 = x_raw_train2[ind_train]
y_train2 = y_raw_train2[ind_train]

# Test data
x_test2 = x_raw_train2[ind_test]
y_test2 = y_raw_train2[ind_test]


#----------------------------------------------------------------
# Subtask 2.1: SVMs with different kernels
#----------------------------------------------------------------


# Function for evaluating a model
def evaluate_svm(model, ndigit=4):
    
    # Kernel used
    kernel = model.kernel
    
    # Fit model
    model.fit(x_train2, y_train2)
    
    # Make predictions
    pred_train = model.predict(x_train2)
    pred_test = model.predict(x_test2)
    
    # Compute accuracy
    acc_train = metrics.accuracy_score(y_train2, pred_train)
    acc_test = metrics.accuracy_score(y_test2, pred_test)
    
    # Print accuracy
    print("Accuracy on training data with", kernel, "kernel:", np.round(acc_train, ndigit))
    print("Accuracy on test data with", kernel, "kernel:", np.round(acc_test, ndigit))
    
    return acc_train, acc_test



# Different kernels with default parameters
svm_linear = svm.SVC(kernel="linear")
svm_poly = svm.SVC(kernel="poly")
svm_rbf = svm.SVC(kernel="rbf")
svm_sigmoid = svm.SVC(kernel="sigmoid")

# Evaluate
evaluate_svm(svm_linear)
evaluate_svm(svm_poly)
evaluate_svm(svm_rbf)
evaluate_svm(svm_sigmoid)


# Selecting good hyperparameters


# --- LINEAR KERNEL ---

# Linear kernels have no additional hyperparameters

best_linear = svm.SVC(kernel="linear")


# --- POLYNOMIAL KERNEL ---

svm_poly = svm.SVC(kernel="poly")

# Degree & coef0

# Parameter grid to search
degrees = [2, 3, 4, 5]
coefs0 = np.logspace(-2, 1, 4)
param_poly = dict(degree=degrees, coef0=coefs0)

# Try different parameters
grid_poly = model_selection.GridSearchCV(svm_poly, param_grid=param_poly, cv=5)
grid_poly.fit(x_train2, y_train2)

# Parameters with the best results
best_params_poly = grid_poly.best_params_
print("Poly kernel best parameters", best_params_poly)

# Model using best parameters
best_poly = svm.SVC(kernel="poly", degree=best_params_poly["degree"], coef0=best_params_poly["coef0"])


# --- RBF KERNEL ---

svm_rbf = svm.SVC(kernel="rbf")

# Gamma

# Parameter grid to search
gammas = np.logspace(-3, 1, 5)
param_rbf = dict(gamma=gammas)

# Try different parameters
grid_rbf = model_selection.GridSearchCV(svm_rbf, param_grid=param_rbf, cv=5)
grid_rbf.fit(x_train2, y_train2)

# Parameters with the best results
best_params_rbf = grid_rbf.best_params_
print("RBF kernel best parameters", best_params_rbf)

# Model using the best parameters
best_rbf = svm.SVC(kernel="rbf", gamma=best_params_rbf["gamma"])



# --- SIGMOID KERNEL ---

svm_sigmoid = svm.SVC(kernel="sigmoid")

# Gamma & coef0

# Parameter grid to search
gammas = np.logspace(-3, 1, 5)
coefs0 = np.logspace(-3, 0, 3)
param_sigmoid = dict(gamma=gammas, coef0=coefs0)

# Try different parameters
grid_sigmoid = model_selection.GridSearchCV(svm_sigmoid, param_grid=param_sigmoid, cv=5)
grid_sigmoid.fit(x_train2, y_train2)

# Parameters with the best results
best_params_sigmoid = grid_sigmoid.best_params_
print("Sigmoid kernel best parameters", best_params_sigmoid)

# Model using the best parameters
best_sigmoid = svm.SVC(kernel="sigmoid", gamma=best_params_sigmoid["gamma"], coef0=best_params_sigmoid["coef0"])




# Evaluate performances of best models
evaluate_svm(best_linear)
evaluate_svm(best_poly)
evaluate_svm(best_rbf)
evaluate_svm(best_sigmoid)





#----------------------------------------------------------------
# Subtask 2.2: Develop your own kernel
#----------------------------------------------------------------




#----------------------------------------------------------------