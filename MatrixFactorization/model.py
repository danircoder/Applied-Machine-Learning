# coding=utf-8
"""
Part 2. The Model
-----------------
Recall that when using Matrix Factorization with biases, our model for prediction is given by:
    rˆ_m,n = µ + u_mT*v_n + b_n + b_m
    
further our loss function is given by:
    1/2*sigma_(m,n)∈I (r_m,n − (µ + u_mT*v_n + bn + bm))^2 + λv/2*sigma_n(||v_n||^2) + λu/2*sigma_m(||u_m||^2) + λb_u/2*sigma_m(b_m^2) + λb_v/2*sigma_m(b_n^2)
    
I - our dataset of ratings
µ - the average rating of all data in data set I
r_m,n - the rating given by user m to item n
u_m, v_n - vector valued parameters representing the “taste” of user m and item n, respectively
b_m, b_n - scalar valued parameters representing the “bias” for user m and item n, respectively
λ - regularization hyper parameters
"""

import numpy as np


class HyperParameters(object):
    """
    Section 1.
    ----------
    Define all hyperparameters required by the model and implement a class to encapsulate these hyperparameters
    """
    def __init__(self,
                 lambda_user=0.0,
                 lambda_item=0.0,
                 lambda_bias_user=0.0,
                 lambda_bias_item=0.0,
                 latent_factor=10):
        """
        Initialization method for the base hyper parameters class
        :param lambda_user: Regularization value for the users
        :param lambda_item: Regularization value for the items
        :param lambda_bias_user: Regularization value for the user bias
        :param lambda_bias_item: Regularization value for the item bias
        :param latent_factor: The number of latent dimensions of the vectors (d)
        """
        self.lambda_user = lambda_user
        self.lambda_item = lambda_item
        self.lambda_bias_user = lambda_bias_user
        self.lambda_bias_item = lambda_bias_item
        self.latent_factor = latent_factor


class MFModel(object):
    """
    Section 2.
    ----------
    Define a class called MFModel that represents all the model parameters. This class should also initialize
    parameter values
    """
    def __init__(self, ratings, hyper, scale=0.01):
        """
        Initialize model object and all it's variables and vectors
        :param ratings: The ratings data set
        :param hyper: Hyper parameters object of the model
        :param scale: The scale initialization parameter for the model vectors
        """
        self.ratings = ratings
        self.hyper = hyper
        if self.ratings.any():
            self.n_users, self.n_items = self.ratings.shape
            ratings_rows, ratings_cols = self.ratings.nonzero()
            self.ratings_indexes = zip(ratings_rows, ratings_cols)
            # randomly initialize
            self.user_vectors = np.random.normal(scale=scale, size=(self.n_users, hyper.latent_factor))
            self.item_vectors = np.random.normal(scale=scale, size=(self.n_items, hyper.latent_factor))
            self.user_bias = np.zeros(self.n_users)
            self.item_bias = np.zeros(self.n_items)
            self.mui = np.mean(self.ratings[np.where(self.ratings != 0)])

    def __get_ratings_prediction(self, ratings):
        """
        Calculate the ratings prediction matrix for all the users and items.
        :return: The prediction matrix according to the model data
        """
        n_users, n_items = ratings.shape
        ratings_rows, ratings_cols = ratings.nonzero()
        ratings_indexes = zip(ratings_rows, ratings_cols)
        ratings_prediction = np.zeros(shape=(n_users, n_items))
        for user, item in ratings_indexes:
            ratings_prediction[user, item] = self.predict(user, item)
        return ratings_prediction

    def predict(self, user, item):
        """
        Predict the ratings for a single item for a user
        :param user: The user index
        :param item: The item index
        :return: The rating prediction for the user and item
        """
        prediction = self.user_vectors[user, :].dot(self.item_vectors[item, :].T)
        prediction += self.user_bias[user] + self.item_bias[item] + self.mui
        return prediction

    def loss_error(self, ratings):
        """
        Calculate the Loss function value for the given ratings matrix parameter.
        Formula:
        --------
        E(Θ) = 1/2*sigma_(m,n)∈I (r_m,n − (µ + u_mT*v_n + bn + bm))^2 + λv/2*sigma_n(||v_n||^2) + λu/2*sigma_m(||u_m||^2) + λb_u/2*sigma_m(b_m^2) + λb_v/2*sigma_m(b_n^2)
        :param ratings: Ratings matrix to check loss value against
        :return: The loss value
        """
        predictions = self.__get_ratings_prediction(ratings)
        error = ratings - predictions
        error = np.power(error, 2).sum()
        error += self.hyper.lambda_item * (np.power(self.item_vectors, 2).sum())
        error += self.hyper.lambda_user * (np.power(self.user_vectors, 2).sum())
        error += self.hyper.lambda_bias_item * (np.power(self.item_bias, 2).sum())
        error += self.hyper.lambda_bias_user * (np.power(self.user_bias, 2).sum())
        error /= 2
        return error

    def squared_error(self, ratings):
        """
        Calculate the SE value for the given ratings matrix parameter.
        Formula:
        --------
        SE = sigma_(m,n)∈I (rm,n − rˆm,n)^2
        :param ratings: Ratings matrix to check MSE against
        :return: The MSE value
        """
        predictions = self.__get_ratings_prediction()
        error = ratings - predictions
        error = np.power(error, 2)
        error = error.mean()  # sum, divided by shape
        return error

    def mean_squared_error(self, ratings):
        """
        Calculate the MSE value for the given ratings matrix parameter.
        Formula:
        --------
        MSE = 1/|T| * sigma_(m,n)∈I (rm,n − rˆm,n)^2
        :param ratings: Ratings matrix to check MSE against
        :return: The MSE value
        """
        predictions = self.__get_ratings_prediction(ratings)
        error = ratings - predictions
        error = np.power(error, 2)
        mse = error.sum() / ratings.nonzero()[0].shape[0]
        return mse
