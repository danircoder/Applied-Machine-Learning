# coding=utf-8
"""
Part 1. The Data
----------------
In this part you are required to :
1. implement a function that reads in the data (path to the csv file is a parameter), removes the Id field and removes entries for which the
   Saleprice attribute is not known.
2. implement a function that splits the data into training and test, the split should be 80% to the Train set and 20% to the Test set.
3. implement a class that encapsulates the training data. The constructor to this class should be given a subset of the data
4. The class above should code categorical features using the following scheme: For each possible value (including the ‘missing’ value) of
   feature i compute the average SalePrice (or label value in general). Next rank the values according to the average computed value (ascending)
   and code the features using the rank.
   The coding map should be saved for use on the test set.
5. The class above should deal with missing numerical features using mean imputation. That is, for each numerical feature compute the average
   value over the training data excluding missing values. Then use this calculated value to fill in the missing values.
   The imputation map should be saved for use on the test set.
6. implement a class encapsulating the test set. The constructor to this class should take as input a subset of the data as well as an imputation
   map and a categorical coding map.
   Categorical features should be coded using the coding map, and missing numeric features should be encoded using the imputation map.
"""

import numpy as np
import pandas as pd


def read_dataset(file_path):
    """
    Reads in the data (path to the csv file is a parameter).
    Removes the Id field and removes entries for which the Saleprice attribute is not known.
    :param file_path: File path to the CSV dataset
    :return: Parsed data frame object
    """
    data_frame = pd.read_csv(file_path, header=0)
    data_frame.info()
    data_frame = data_frame.drop(labels=["Id"])
    data_frame = data_frame.dropna(axis=0, subset=["SalePrice"])
    data_frame.info()
    return data_frame


def split_dataset_train_test(data_frame):
    """
    Splits the data into training and test, the split should be 80% to the Train set and 20% to the Test set
    :param data_frame: Input data frame to split
    :return: Two data frames (Train, Test)
    """
    masking = np.random.rand(len(data_frame))
    train = data_frame[masking]
    test = data_frame[~masking]
    return train, test


class Train:
    """
    Training dataset encapsulating class
    """
    def __init__(self, data_frame, label_feature_name):
        """
        Initialize the train data frame
        :param data_frame: Data frame object with the training subset
        :param label_feature_name: The label feature name
        """
        self.data_frame = data_frame
        self.label_name = label_feature_name
        columns_dictionary = dict(data_frame.dtypes)
        self.features = [feature for feature in columns_dictionary.keys() if feature != self.label_name]
        self.categorical_features = [feature for feature, feature_type in columns_dictionary.iteritems() if feature != self.label_name and feature_type == "object"]
        self.numerical_features = [feature for feature, feature_type in columns_dictionary.iteritems() if feature != self.label_name and feature_type != "object"]
        self.categorical_map_dict = {}
        self.numerical_imputation_value_dict = {}

    def __create_categorical_features_map(self):
        """
        Create a categorical features map by iterating each unique value in a categorical feature, and calculate the mean value of the
        label column for that subset and rank the values in ascending order and then create a dictionary for each value and it's rank.
        """
        for feature in self.categorical_features:
            unique_values = self.data_frame[feature].unique()
            values_tuples = []
            for value in unique_values:
                condition = self.data_frame[feature] == value
                value_label_mean = self.data_frame[condition][self.label_name].mean()
                values_tuples.append((value, value_label_mean))
            values_tuples.sort(key=lambda tuple_value: tuple_value[1])
            self.categorical_map_dict[feature] = {}
            rank = float(1.0)
            for value in values_tuples:
                self.categorical_map_dict[feature][value[0]] = rank
                rank += 1.0

    def __create_numerical_features_map(self):
        """
        Create a numberical features map by iterating each numerical feature and calculate it's mean value (without N/A) and set that
        value as the impute value for the feature.
        """
        for feature in self.numerical_features:
            mean_value = self.data_frame[feature].mean(skipna=True)
            self.numerical_imputation_value_dict[feature] = mean_value

    def handle_categorical_features(self):
        """
        For each possible value (including the ‘missing’ value) of feature i compute the average SalePrice (or label value in general).
        Than rank the values according to the average computed value (ascending) and code the features using the rank.
        """
        # Create categorical features rank mapping
        self.__create_categorical_features_map()

        # Iterate each categorical feature and replace it's values with a rank
        for feature in self.categorical_features:
            for value, rank in self.categorical_map_dict[feature].iteritems():
                self.data_frame[feature].replace(to_replace=value, value=rank, inplace=True)

    def handle_numerical_imputation(self):
        """
        Deal with missing numerical features using mean imputation.
        For each numerical feature compute the average value over the training data excluding missing values.
        Then use this calculated value to fill in the missing values.
        """
        # Create numerical features imputation map
        self.__create_numerical_features_map()

        # Iterate each numerical feature and fill n/a values with the mean
        for feature in self.numerical_features:
            self.data_frame[feature].fillna(value=self.numerical_imputation_value_dict[feature], inplace=True)


class Test:
    """
    Testing dataset encapsulating class
    """
    def __init__(self, data_frame):
        """
        Initialize the test data frame
        :param data_frame: Data frame object with the test subset
        """
        self.data_frame = data_frame

    def handle_features(self, categorical_map, numerical_map):
        """
        Normalize and impute missing values according to categorical and numerical values.
        :param categorical_map: Categorical features ranking map for all unique values including nan
        :param numerical_map: Numerical features imputation map according to mean value
        """
        # Iterate each categorical feature and replace it's values with a rank
        for feature in categorical_map:
            for value, rank in categorical_map[feature].iteritems():
                self.data_frame[feature].replace(to_replace=value, value=rank, inplace=True)

        # Iterate each numerical feature and fill n/a values with the mean
        for feature in numerical_map:
            self.data_frame[feature].fillna(value=numerical_map[feature], inplace=True)
