"""
Part 1. The Data
----------------
Download and unzip the movie lens 1M dataset:
A description of the different files can be found in README.
For our purposes we will make use of ratings.dat and movies.dat.

http://files.grouplens.org/datasets/movielens/ml-1m.zip
"""

import os
import pandas as pd
import numpy as np
from abc import ABCMeta


class BaseDataReader(object):
    """
    Base data reader for *.dat files from the movie lens 1M dataset.
    """
    __metaclass__ = ABCMeta

    def __init__(self, data_file_path, columns_names):
        """
        Initialization method to load dataset file into a DataFrame object
        :param data_file_path: Path to dataset file
        """
        if os.path.isfile(data_file_path):
            self.headers = columns_names
            self.data = pd.read_csv(filepath_or_buffer=data_file_path,
                                    sep='\:\:',
                                    lineterminator='\n',
                                    header=None,
                                    names=columns_names)
        else:
            raise IOError("File not found {0}".format(data_file_path))

    def head(self, count):
        """
        Print the DataFrame structure with count records received as argument
        :param count: Records count to print
        """
        print ""
        print self.data.head(count)

    def info(self):
        """
        Print the DataFrame structure information
        """
        print ""
        print self.data.info()


class Ratings(BaseDataReader):
    """
    Section 1.
    ----------
    Implement a data structure to store ratings data with a constructor that takes the path to ratings.dat
    as an argument
    """
    def __init__(self, data_file_path):
        """
        Initialization method to load dataset file into a DataFrame object
        :param data_file_path: Path to dataset file
        """
        super(Ratings, self).__init__(data_file_path, ['UserID', 'MovieID', 'Score', 'Timestamp'])
        self.matrix = self.data.pivot(index="UserID", columns="MovieID", values="Score").fillna(0).as_matrix()

    def get_matrix(self):
        """
        Getter function to return the matrix object
        :return: Sparse matrix of movie ratings by users
        """
        return self.matrix

    def split_dataset(self, test_percent=0.2):
        """
        Splits the sparse matrix into a train and test objects which keeps an instance of each user in the train and
        test matrices and only splits the rating values
        :param test_percent: Percentage of the test data set (80% train, 20% test)
        :return: Two matrices of train and test from the sparse matrix
        """
        train = self.matrix.copy()
        test = np.zeros(self.matrix.shape)
        for user in xrange(self.matrix.shape[0]):
            nonzero_in_row = self.matrix[user, :].nonzero()[0]
            if nonzero_in_row.shape[0] > 0:
                random_item = np.random.choice(nonzero_in_row, 1)
                random_index = np.argwhere(nonzero_in_row == random_item[0])
                if nonzero_in_row.shape[0] > 1:
                    nonzero_in_row = np.delete(nonzero_in_row, random_index)
                test_samples_in_row = np.random.choice(a=nonzero_in_row,
                                                       size=int(test_percent * len(nonzero_in_row)),
                                                       replace=False)
                test_samples_in_row = np.append(test_samples_in_row, random_item)
                train[user, test_samples_in_row] = 0.
                test[user, test_samples_in_row] = self.matrix[user, test_samples_in_row]
        return train, test

    @staticmethod
    def show_sparsity(matrix, title):
        """
        Calculate the sparsity ratio of the matrix, and print the value with a title string format
        :param matrix: The matrix object to calculate sparsity
        :param title: Title format string for the print of the sparsity value
        """
        sparsity = float(len(matrix.nonzero()[0]))
        sparsity /= (matrix.shape[0] * matrix.shape[1])
        sparsity *= 100
        print ""
        print title + 'Sparsity: {:4.2f}%'.format(sparsity)


class Movies(BaseDataReader):
    """
    Section 1.
    ----------
    Implement a data structure to store the item data with a constructor that takes the path to movies.dat
    as an argument
    """
    def __init__(self, data_file_path):
        """
        Initialization method to load dataset file into a DataFrame object
        :param data_file_path: Path to dataset file
        """
        super(Movies, self).__init__(data_file_path, ['MovieID', 'Title', 'Genres'])
        self.dictionary = None

    def get_dictionary(self):
        """
        Getter function to return the dictionary object
        :return: Dictionary with all the movies with their index as key
        """
        if self.dictionary:
            return self.dictionary
        self.dictionary = {}
        for index, row in self.data.iterrows():
            self.dictionary[row['MovieID']] = (row['Title'], row['Genres'])
        return self.dictionary

