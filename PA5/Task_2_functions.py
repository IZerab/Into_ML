# this is a lib containing the useful function for task 2 PA 5
# author: Lorenzo Beltrame

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split


def initialize_data():
    """
    This function downloads the dataset Labeled Faces in the Wild if it has not already been downloaded.
    It returns the dataset already initialised as asked in the assignment.
    :return: the initialized Labeled Faces in the Wild dataset already splitted into train/test and stratified
    """
    # load the scikit dataset object (a cache is set!)
    df_obj = fetch_lfw_people(data_home='./Dataset_2/', min_faces_per_person=30, resize=0.5)

    # get the dataframe
    df = pd.DataFrame(df_obj.data)
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

    return X_train, X_test, y_train, y_test


def explore_data(dataset_object):
    """
    This function answers the question of subtask 2.1 of the PA.
    □ How many different people are in the data?
    □ How many images are in the data?
    □ What is the size of the images?
    □ Plot images of ten different people in the data set.
    :return:
    """

    num_people = len(dataset_object.target_names)
    print("There are {} different people in the dataset's pictures".format(num_people))

    num_images = dataset_object.data.shape[0]
    size_images = [dataset_object.images.shape[1], dataset_object.images.shape[2]]
    print("There are {} different pictures of shape {}".format(num_images, size_images))

    # plot images of 10 different people
    plt.figure(figsize=(10, 12))

    unique_names = pd.Series(dataset_object.target).drop_duplicates()
    index_unique = unique_names.index[:9]

    plt.subplots_adjust(0.6, 0.5, 1.5, 1.5)

    # convenience variable
    position = 0
    for i in index_unique:
        plt.subplot(1, 10, position + 1)
        plot_image(dataset_object.images[i], subplot=True)
        position += 1
    plt.tight_layout()

    print("\nThe first ten pictures are associated with:")
    for i in index_unique:
        print(dataset_object.target[i])


def plot_image(image, subplot=False):
    """
    Function that plots the image passed into the notebook.
    :param subplot: if true it need a subplot structure to print!
    :param image: image to plot
    """
    if not subplot:
        plt.figure(figsize=(3, 4))

    plt.imshow(image, cmap=plt.cm.gray)
    plt.xticks(())
    plt.yticks(())


class my_pca:
    """
    This function implements principal component analysis.
    It includes 4 methods.
    """
    def __init__(self):
        """
        Initialize the class and create useful variables to store the information.
        """
        self.
