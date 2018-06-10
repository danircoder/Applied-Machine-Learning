"""
Download and unzip the Stanford Treebank data (1) - http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip.
The file README.txt describes the different files included.
For our purposes we will use only dataSetSentences.txt and datasetSplit.txt.
In this part you are required to :
1. implement a class to parse the sentence assignments to training and test sets. The constructor of the class should take as an argument the
   path to datasetSplit.txt.
2. implement a class to hold the training/test data.
   The constructor of the class should receive the path to dataSetSentences.txt and an instance of the class implemented above (to indicate assignment
   of sentences to train/test).
   Each sentence should be stored as an (ordered) list of tokens which are the words (split on whitespace) of the sentence after the following preprocessing:
      (a) lowercase
      (b) removal of non-ascii characters
      (c) removal of non-alphanumeric characters
      (d) removal of words with less than 3 characters
"""

import os
import re
import pandas as pd
from abc import ABCMeta


class BaseDataReader(object):
    """
    Base data reader for corpus sentences dataset files.
    """
    __metaclass__ = ABCMeta

    def __init__(self, file_path, separator):
        """
        Initialization method to load dataset file into a DataFrame object
        :type separator: Separator character for the file
        :param file_path: Path to dataset file
        """
        if os.path.isfile(file_path):
            self.path = file_path
            self.data = pd.read_csv(filepath_or_buffer=file_path,
                                    sep=separator,
                                    lineterminator=None,
                                    header=0)
        else:
            raise IOError("File not found {0}".format(file_path))


class CorpusSplit(BaseDataReader):
    """
    Section 1:
    ----------
    Implement a class to parse the sentence assignments to training and test sets. The constructor of the class should take as an argument the
    path to datasetSplit.txt.
    """
    def __init__(self, file_path):
        """
        Initialization method to load dataset split file into a DataFrame object
        :param file_path: Path to dataset split file
        """
        super(CorpusSplit, self).__init__(file_path, ',')
        self.dictionary = dict(zip(self.data.sentence_index, self.data.splitset_label))


class CorpusSentences(BaseDataReader):
    """
    Section 2:
    ----------
    Implement a class to hold the training/test data. The constructor of the class should receive the path to dataSetSentences.txt and an instance
    of the class implemented above (to indicate assignment of sentences to train/test).
    Each sentence should be stored as an (ordered) list of tokens which are the words(split on whitespace) of the sentence after the following
    preprocessing:
        (a) lowercase
        (b) removal of non-ascii characters
        (c) removal of non-alphanumeric characters
        (d) removal of words with less than 3 characters
    """
    def __init__(self, file_path, split):
        """
        Initialization method to load dataset split file into a DataFrame object
        :type split: Instance of the CorpusSplit class
        :param file_path: Path to dataset split file
        """
        super(CorpusSentences, self).__init__(file_path, '\t')
        dictionary = dict(zip(self.data.sentence_index, self.data.sentence))
        self.train = []
        self.test = []
        self.other = []
        self.pattern = re.compile('[\W_]+')
        for index, sentence in dictionary.iteritems():
            if index == 0:
                continue
            if split.dictionary[index] == 1:
                self.train.append(self.__preprocess_sentence(sentence))
            elif split.dictionary[index] == 2:
                self.test.append(self.__preprocess_sentence(sentence))
            else:
                self.other.append(self.__preprocess_sentence(sentence))

    def __preprocess_sentence(self, sentence):
        processed_tokens = []
        tokens = sentence.split(" ")
        for token in tokens:
            token = token.lower()
            token = "".join(char for char in token if ord(char) < 128)
            token = self.pattern.sub('', token)
            if len(token) < 3:
                continue
            processed_tokens.append(token)
        return processed_tokens
