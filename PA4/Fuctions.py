# Those are useful functions for the project Programming assignment 4
# author: Lorenzo Beltrame

import pandas as pd
import numpy as np
import csv


def import_movie_data():
    """
    This function import movie data from the files present in the folder, be sure that they are present!
    Note that the separator of the .dat files is "::" and you need to pass tha names of the columns
    :return: 3 pandas dataframes: df_movies, df_ratings, df_users
    """
    # I manually insert the column names
    col_movies = ["MovieID", "Title", "Genres"]
    col_ratings = ["UserID", "MovieID", "Rating", "Timestamp"]
    col_users = ["UserID", "Gender", "Age", "Occupation", "Zip-code"]

    # read flash.dat to a list of lists
    df_movies = pd.read_table('movies.dat', sep=r"::", header=None, engine="python", names=col_movies)
    df_ratings = pd.read_table('ratings.dat', sep=r"::", header=None, engine="python", names=col_ratings)
    df_users = pd.read_table('users.dat', sep=r"::", header=None, engine="python", names=col_users)
    return df_movies, df_ratings, df_users


def create_dataframe(df_movies, df_ratings, df_users):
    """
    This function performs a series of "Pandas.merge" methods to create a dataframe in which each entry is a review
    with the informations of the related film
    :param df_movies: dataframe containing info about the movies
    :param df_ratings: dataframe containing info about the ratings
    :param df_users: dataframe containing info about the users
    :return: dataframe to be used in the ML
    """
    # check if they are all pandas DF
    inputs = [df_movies, df_ratings, df_users]
    for i in inputs:
        if not isinstance(i, pd.DataFrame):
            raise ValueError("Value error: The input is not a pandas DF!")

    # the base for our joining in df_ratings!
    df_ratings = df_ratings.merge(df_movies, on="MovieID", how="left")
    df = df_ratings.merge(df_users, on="UserID", how="left")

    return df


def remove_users(df, N):
    """
    Remove all users which have rated less than N movies.
    :param df: pandas dataframe that includes movies
    :param N: threshold of movies watched
    :return:
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Value error: The input is not a pandas DF!")

    temp1 = df.shape[0]

    # group by the user feature (they have the same index)
    groups = df.groupby(by="UserID").count()[lambda x: x >= N]
    groups.dropna(inplace=True)
    df = df[df["UserID"].isin(groups.index)]
    temp2 = df.shape[0]
    print("The number of reviews dropped is: ", temp1 - temp2)

    return df


def binarizer(x):
    """
    Rule to binarize the data. See binarize rating for more
    :param x: convenience variable
    :return: x binarized
    """
    if x == 5 or x == 4:
        # binarize to 1
        x = 1
    else:
        # binarize to 0
        x = 0
    return x


def binarize_rating(df):
    """
    The ratings are given as 1-star to 5-star ratings. Convert these ratings to binary
    labels, such that 4 and 5 stars ratings are mapped to label “1” and 1, 2, and 3 stars
    ratings are mapped to label “0”. We are using the convenience function binarize.
    :param df: Pandas dataframe containing the aggregated movie data
    :return: the dataframe with the binarized columns
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Value error: The input is not a pandas DF!")

    # I create a copy since we are going to slice!
    temp_df = df.copy()
    temp_df["Rating"] = temp_df["Rating"].apply(binarizer)

    return temp_df


def custom_train_test(df):
    """
    Custom train test split: the test are all the ratings of users with user ids 1, . . . , 1000
    (unless a user was removed). The remaining are the train.
    :param df: Pandas dataframe containing the aggregated movie data
    :return: train and test pandasDF
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Value error: The input is not a pandas DF!")
    temp_df = df.copy()
    train = temp_df[temp_df["UserID"] > 1000]
    test = temp_df[temp_df["UserID"] <= 1000]
    return train, test
