# coding=utf-8

"""
Recall from lecture that under the skip-gram model we are trying to maximize the probability of “context words” given the “input word”.
Consider the sentence Gal Loves IDC. Using a context window of size 1 (on each side of the input), would yield the context word combinations as shown
in the images.

In the negative-sampling variant of word2vec the (log) probability of seeing a context word in the dataset, denoted w_c below, given an input word,
denoted w_i, is given by the expression:
log P (w_c | w_i) = log σ(u_iTv_c) - simga_k=1^K log (1 − σ(u_iTv_k))

where u_i ∈ R^d and vi ∈ R^d are the ‘target’ and ‘context’ representation of each word in our vocabulary.
These are our model parameters, collectively denoted Θ.
Θ = {{u_i}_i=1^|V|, {v_i}_i=1^|V|}

Equation 1 also assumes that K ‘non-context’ words, w_1, . . . , w_K are sampled from some distribution over the vocabulary, V.
In this part you are required to:
1. implement a class representing the model hyperparameters which include: the size of the context window, the size of the vector representation of
   each word, the number of “noncontext” negative words, the number of iterations between reductions of the learning rate, the choice of noise distribution,
   and random seed governing randomness in initialization (to allow reproducibility).
2. implement a class that represents your model parameters. This class should take as input an instance of the hyperparameters class
3. implement a member function Init, that takes the training data as input, to initialize the parameters as follows:
   (a) Create a vector for each word in the training data
   (b) Sample each vector from a multivariate Gaussian with mean 0 (vector) and Covariance 0.01 · ID (the Identity matrix)
   (c) Normalize the random vector to be unit length (using the L2 norm)
4. implement a function to sample words from the distribution:
   P(w) ∝ U(w)^α
   where U(w) is the unigram distribution and 0 ≤ α ≤ 1 is a parameter.
   The unigram distribution should be calculated based on the training data.
5. implement a function to sample K random words from the vocabulary using the noise distribution specified in the hyperparameters
6. implement a function to return the log probability of context word i given input word j and K negative samples.
"""

import numpy as np
from collections import defaultdict


class ModelHyperParameters(object):
    """
    Section 1:
    ----------
    implement a class representing the model hyperparameters which include: the size of the context window, the size of the vector representation of
    each word, the number of “noncontext” negative words, the number of iterations between reductions of the learning rate, the choice of noise distribution,
    and random seed governing randomness in initialization (to allow reproducibility).
    """
    def __init__(self,
                 C=1,
                 d=3,
                 K=3,
                 alpha=(2.0/3.0),
                 seed=1.0):
        """
        Initialization method for the base hyper parameters class
        :param C: the size of context window
        :param d: the size of the vector representation of each word
        :param K: the number of "non-context" negative words
        :param alpha: the choice of noise distribution
        :param seed: random seed governing randomness in initialization
        """
        self.C = C
        self.d = d
        self.K = K
        self.alpha = alpha
        self.seed = seed


class SkipGramModel(object):
    """
    Section 2.
    ----------
    implement a class that represents your model parameters. This class should take as input an instance of the hyperparameters class.
    """
    def __init__(self, hyper):
        """
        Initialize model object and all it's variables and vectors
        :param hyper: Hyper parameters object of the model
        """
        self.hyper = hyper
        self.dataset = None
        self.V = None
        self.u = None
        self.v = None
        self.V_unigram = None
        self.V_probs = None
        self.P_sentences = None

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))

    def init(self, dataset):
        """
        Section 3.
        ----------
        implement a member function Init, that takes the training data as input, to initialize the parameters as follows:
            (a) Create a vector for each word in the training data
            (b) Sample each vector from a multivariate Gaussian with mean 0 (vector) and Covariance 0.01 · ID (the Identity matrix)
            (c) Normalize the random vector to be unit length (using the L2 norm)
        :param dataset: The training dataset for the model
        :return: V - vocabulary of the training data.
        """
        # save pointer to the training set
        self.dataset = dataset

        # create V - Vocabulary from the training data:
        # V['some_word'] = {(row, word), (row, word), (row, word), (row, word)}
        self.V = self.create_vocabulary(dataset)

        # create the vector representation for each word
        # u['some_word'] = [0.01 -0.01 0.2] which is a ndarray
        # v['some_word'] = [0.01 -0.01 0.2] which is a ndarray
        self.u, self.v = self.__create_words_vectors()

        # create the vocabulary distribution probabilities and probabilities sum
        self.V_unigram = self.get_unigram_alpha_dict(self.V)
        self.V_probs = self.get_vocabulary_probabilities(self.V_unigram)

        # create corpus probability to sample target word
        self.P_sentences = self.get_sentences_probabilities(dataset)

    @staticmethod
    def get_vocabulary_probabilities(V_unigram):
        """
        Normalize the probabilities in order to sum to 1
        :param V_unigram: The unigram vocabulary distribution
        :return: Normalized probabilities of the vocabulary
        """
        probs = np.array(V_unigram.values())
        probs /= probs.sum()
        return probs

    @staticmethod
    def get_sentences_probabilities(dataset):
        # create P_rows[row_1] = |#words_in_row1| / |#words_in_corpus|
        P_rows = defaultdict(float)
        words_in_corpus = 0.0
        index_row = 0
        for row in dataset:
            words_in_row = len(row)
            # P_rows[row_1] = |#words_in_row1|
            P_rows[index_row] = words_in_row
            words_in_corpus += words_in_row
            index_row += 1
        for index_row in P_rows.keys():
            # P_rows[row_1] = |#words_in_row1| / |#words_in_corpus|
            P_rows[index_row] /= words_in_corpus
        return P_rows

    @staticmethod
    def create_vocabulary(dataset):
        """
        create a dictionary (using collections.defaultdict) as following:
        V['some_word'] = {(row, word), (row, word), (row, word), (row, word)}
        :param dataset: The training dataset for the model
        :return: V - vocabulary of the training data.
        """
        # V['some_word'] = {(row, word), (row, word), (row, word), (row, word)}
        V = defaultdict(list)
        index_row = 0
        for row in dataset:
            index_word = 0
            for word in row:
                V[word].append((index_row, index_word))
                index_word += 1
            index_row += 1
        return V

    def __create_words_vectors(self):
        """
        Creates the vector representation for each word as following:
        for d=3: words_vectors['some_word'] = [0.01 -0.01 0.2] which is a ndarray
        :return: Words Vectors (u, v) - default dict of the vectors representing the words.
        """
        u = defaultdict(np.ndarray)
        v = defaultdict(np.ndarray)
        for word in self.V.iterkeys():
            # Sample each vector from a multivariate Gaussian with mean 0 (vector) and Covariance 0:01 · ID (the Identity matrix)
            # random.normal args: size = dimensions , loc = mean , scale = std = sqrt(Variance) = sqrt(0.01) = 0.1
            rand_vec = np.random.normal(size=int(self.hyper.d), loc=0.0, scale=0.1)

            # Normalize the random vector to be unit length (using the L2 norm)
            l2_norm = np.linalg.norm(x=rand_vec, ord=2, keepdims=True)
            rand_vec_normalized = rand_vec/l2_norm
            u[word] = rand_vec_normalized
            v[word] = rand_vec_normalized
        return u, v

    def get_unigram_alpha_dict(self, V):
        """
        Section 4.
        ----------
        Implement a function to sample words from the distribution: P(w) ∝ U(w)^α
        Where U(w) is the unigram distribution and 0 ≤ α ≤ 1 is a parameter.
        The unigram distribution should be calculated based on the training data.
        This can be done via U(w) = f(w)^α/Sig_w'(f(w')^α);
        THIS FUNC IS USED BY: model.get_negative_samples()
        :param V: Vocabulary from the dataset
        :return: Unigram distribution for the vocabulary
        """
        # calculate frequency for each word, and also sigma_freq_power_alpha = Sigma_foreach_w[freq(w)^α)
        freq_dict = defaultdict(int)
        sigma_freq_power_alpha = 0
        for word, indices_list in V.iteritems():
            freq_dict[word] = len(indices_list)
            sigma_freq_power_alpha += freq_dict[word]**self.hyper.alpha

        # calculate final f(w)^α/Sig_w'(f(w')^α)
        unigram_alpha_dict = defaultdict(float)
        for word, freq in freq_dict.iteritems():
            freq_pow_alpha = (freq**self.hyper.alpha)
            unigram_alpha_dict[word] = freq_pow_alpha / (sigma_freq_power_alpha - freq_pow_alpha)

        return unigram_alpha_dict

    def get_negative_samples(self, V_unigram, V_probs):
        """
        Section 5.
        ----------
        Implement a function to sample K random words from the vocabulary, using the noise distribution specified in the hyperparameters.
        :param V_unigram: Unigram distribution with probabilities of the vocabulary
        :param V_probs: Probabilities of the vocabulary
        :return: Array of K_words
        """
        return np.random.choice(a=V_unigram.keys(), size=self.hyper.K, p=V_probs)

    def get_log_probability(self, w_c, w_t, K_words):
        """
        Section 6.
        ----------
        Implement a function to return the log probability of context word i given input word j and K negative samples.
        # check the formula: log(σ(v_c^T * u_t)) + Sig_k[log(1 − σ(v_c^T * u_t))]
        log P (w_c | w_t) = log(σ(u_t^T * v_c)) + Sig_k[log(1 − σ(u_t^T * v_k))]
        :param w_c: Context word
        :param w_t: Target word
        :param K_words: K non-context words
        :return: Probability value of log P (w_c | w_t)
        """
        part1 = np.log(self.sigmoid(self.v[w_c].T.dot(self.u[w_t])))
        part2 = 0
        for w_k in K_words:
            part2 += np.log(1 - self.sigmoid(self.v[w_k].dot(self.u[w_t])))
        log_probability = part1 + (1 / len(K_words)) * part2  # TODO: added 1/K fix
        return log_probability

    def sample_target_word(self):
        """
        Randomly choose a target word from teh corpus by the words distribution (unigram).
        :return: Target word
        """
        # sample a row according to P_rows
        sampled_row = np.random.choice(a=self.P_sentences.keys(), size=1, p=np.array(self.P_sentences.values()))

        # sample a word inside the chosen row
        sampled_word = np.random.choice(a=range(len(self.dataset[int(sampled_row)])), size=1)

        # return index tuple: target_word_index = (row, column)
        target_word_index = (int(sampled_row), int(sampled_word))
        return target_word_index
