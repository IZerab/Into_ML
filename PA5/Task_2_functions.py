# this is a lib containing the useful function for task 2 PA 5
# author: Lorenzo Beltrame

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import os
import pickle
# standard libs
from matplotlib.pyplot import plot
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPRegressor
import pickle
import os.path


def initialize_data():
    """
    This function downloads the dataset Labeled Faces in the Wild if it has not already been downloaded.
    It returns the dataset already initialised as asked in the assignment.
    :return: the initialized Labeled Faces in the Wild dataset already splitted into train/test and stratified and the
                design matrix as a whole (for convenience)
    """
    # load the scikit dataset object (a cache is set!)
    df_obj = fetch_lfw_people(data_home='./Dataset_2/', min_faces_per_person=30, resize=0.5)

    # get the dataframe
    df = pd.DataFrame(df_obj.data).copy()
    target = pd.Series(df_obj.target)

    # explore the data
    explore_data(df_obj)

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        df,
        target,
        stratify=target,
        random_state=42,
        test_size=0.3)

    return X_train, X_test, y_train, y_test, df


def centre_data(X):
    """
    Centers the data st the columns have average 0
    :param X: design matrix
    :return: centered data and the mean
    """
    X_mean = X.mean(axis=0)
    X -= X_mean
    return X, X_mean


def explore_data(dataset_object):
    """
    :param dataset_object: sklearn dataset object
                           (https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_people.html)
    This function answers the question of subtask 2.1 of the PA.
    □ How many different people are in the data?
    □ How many images are in the data?
    □ What is the size of the images?
    □ Plot images of ten different people in the data set.
    """

    num_people = len(dataset_object.target_names)
    print("There are {} different people in the dataset's pictures".format(num_people))

    num_images = dataset_object.data.shape[0]
    size_images = [dataset_object.images.shape[1], dataset_object.images.shape[2]]
    print("There are {} different pictures of shape {}".format(num_images, size_images))

    # plot images of 10 different people
    plt.figure(figsize=(10, 12))

    unique_names = pd.Series(dataset_object.target).drop_duplicates()
    index_unique = unique_names.index[:10]

    # plot
    plt.subplots_adjust(0.6, 0.5, 1.5, 1.5)
    print("I visualize ten different pictures presented in the dataset:")
    plot_gallery(dataset_object.images[index_unique], 10)

    print("\nThe first ten pictures are associated with:")
    for i in index_unique:
        print(dataset_object.target[i])


def plot_image(image, subplot=False, shape=None):
    """
    Function that plots the image passed into the notebook.
    :param shape: shape of the image, [62, 47] is the shape of this project!
    :param subplot: if true it need a subplot structure to print!
    :param image: image to plot
    """
    if shape is None:
        shape = [62, 47]

    if not subplot:
        plt.figure(figsize=(3, 4))

    plt.imshow(np.real(image).reshape(shape), cmap=plt.cm.gray)
    plt.xticks(())
    plt.yticks(())


def plot_gallery(images, num=None, shape=None):
    """
    Function that plots all the the images passed to it into the notebook.
    :param shape: shape of the image, [62, 47] is the shape of this project!
    :param num: Number of images to plot, if not specified the lenght of the list is used.
    :param images: list of images to plot
    """
    # plot images of 10 different people
    plt.figure(figsize=(10, 12))

    if num is None:
        K = len(images)
    else:
        K = num
    # convenience variable
    position = 0
    for i in range(K):
        plt.subplot(1, K, position + 1)
        plot_image(images[i], subplot=True, shape=shape)
        position += 1
    plt.tight_layout()
    plt.show()


class my_custom_pca:
    """
    This function implements principal component analysis for images!
    It includes 3 features:
    □ Compute the Principal components
    □ Project given data onto the principal components
    □ Reconstruct images from the projection onto the principal components
    """

    def __init__(self):
        """
        Initialize the class and create useful variables to store the information.
        """
        self.principal_components = None
        # Linear transformation weight
        self.W = None
        # projections of the principal components
        self.projections = None
        # reconstructions
        self.reconstructions = None
        # mean
        self.mean = None

    def fit(self, X, K, mean):
        """
        Performs the fit to compute the principal components.
        The principal components are stored inside the class.
        :param X: design matrix used to train, pandas df, needs to be centered
        :param K: number of eigenvectors to choose
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Insert a pandas DataFrame")

        # save mean
        self.mean = mean

        # Compute the covariance matrix
        S = X.cov()

        # Compute eigenvectors and eigenvalues of the covariance matrix
        eigen_values, eigen_vectors = np.linalg.eig(S)

        # Sort the eigenvalues from highest to lowest
        sort_perm = eigen_values.argsort()[::-1]
        eigen_values.sort()
        eigen_vectors = eigen_vectors[:, sort_perm]

        # save the computed principal components
        self.principal_components = eigen_vectors.T

        # choose K eigenvectors (transpose is necessary for conventional issues!)
        self.W = eigen_vectors[:, :K]

    def transform(self, X):
        """
        This function transform the passed designed matrix into the new subspace
        :param X: design matrix, data are centered
        :return: ndarray, the trasformed data of dimension K, where K
        """
        # due to numerical errors we might have imaginary parts!!
        self.projections = np.real(self.W.T @ X.T)
        return self.projections

    def reconstruct_images_vectors(self):
        """
        This function reconstruct images from the projection onto the principal components.
        :return: ndarray, the reconstructions
        """
        if self.projections is None:
            raise ValueError("Train the PCA first!")

        self.reconstructions = np.real(self.projections.T @ self.W.T)
        for j in self.reconstructions:
            j = j + self.mean

        return self.reconstructions


def encode(X, mlp):
    """
    This function is not working for general MLPs,
    the MLP must have the layer-configuration as
    stated in the exercise description.
    X must have the shape
    n_images x (witdh in pixels * height in pixels)
    """

    z = X
    for i in range(len(mlp.coefs_) // 2):
        z = z @ mlp.coefs_[i] + mlp.intercepts_[i]
    z = np.maximum(z, 0)
    return z


def decode(Z, mlp):
    """
    This function is not working for general MLPs,
    the MLP must have the layer-configuration as
    stated in the exercise description.
    Z must have the shape n_images x d
    """

    z = Z
    for i in range(len(mlp.coefs_) // 2, len(mlp.coefs_)):
        z = z @ mlp.coefs_[i] + mlp.intercepts_[i]
    # z = np.maximum(z, 0)
    return z


def custom_autoencoder_analysis(hidden_layer_sizes, X, X_train, X_test, y_train, y_test, model_name):
    """
    This function first checks if the selected autoencoder exists within the folder the main is in, if it is present,
    the autoencoder is imported in the main, otherwise it is computed. The model is fit both on train and test data.
    Subsequently, the data projected by the autoencoder are min_max scaled and used to train a logistic regressor.
    The accuracies for the trainign and for the testing set are printed.
    All the inputs (X, X_train, X_test) should be centered to zero
    :param hidden_layer_sizes: size of the layers on the NN, they must be of shape (a, b, d, b, a).
    :param X: design matrix
    :param X_train: train part of the design matrix
    :param X_test: test part of the design matrix
    :param y_train: target vector for training
    :param y_test: target vector for testing
    :param model_name: path of the model saved in the same folder as the main
    """
    # I saved the trainings for convenience!
    if os.path.isfile(model_name):
        with open(model_name, 'rb') as f:
            nn = pickle.load(f)
    else:
        # initialize my nn
        nn = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=2000)

        # fit my nn with all the data that I have at my disposal (both train and test)
        nn.fit(X, X)

        # save
        with open(model_name, 'wb') as f:
            pickle.dump(nn, f)

    # try my functions!
    Z_projected = encode(X=X, mlp=nn)
    Z_reconstructed = decode(Z=Z_projected, mlp=nn)

    plot_gallery(Z_reconstructed.T, 10)

    # initialize the scaler
    my_scaler = MinMaxScaler()

    # initialize the logistic classifier
    # I decided to use saga as the solver bc "lbsfg" couldn't converge!
    clf = LogisticRegression(random_state=42, max_iter=5000, solver='saga')

    # create the projection of each train/test set with the PCA fitted over all the data
    Z_projected_train = encode(X=X_train, mlp=nn)
    Z_projected_test = encode(X=X_test, mlp=nn)

    # scale the projections
    # fit
    my_scaler.fit(Z_projected)
    # trasform
    Z_projected_train = my_scaler.transform(Z_projected_train)
    Z_projected_test = my_scaler.transform(Z_projected_test)

    # fit
    clf.fit(Z_projected_train, y_train)
    # predict
    predictions_test = clf.predict(Z_projected_test)
    predictions_train = clf.predict(Z_projected_train)
    # accuracy metric
    print("The accuracy score fot the training is: {}".format(accuracy_score(predictions_train, y_train)))
    print("The accuracy score fot the test is: {}".format(accuracy_score(predictions_test, y_test)))
