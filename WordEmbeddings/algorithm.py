# coding=utf-8

"""
In this section we will use stochastic gradient descent(SGD) (actually ascent)to find the setting of model parameters, Θ which maximize the (log) Likelihood:
L(Θ) = sigma_(w_c,w_i)∈D log P (w_c | w_i)

where D is the set of all context/input word pairs in our training data.
Rather than use the entire dataset to compute the gradient we will use a small subset of context/input pairs, selected uniformly at random, called a
mini-batch, to perform the optimization.
Each iteration of SGD will thus use the following update rule:
Θ_new = Θ_old + η(simga_(w_c,w_i)∈D_mb ∇Θ [log P (w_c | w_i)] (Θ_old)

where η is a hyperparameter called the learning rate, and Dmb is a mini-batch of context/input pairs.
In this part you are required to:

1. Derive the gradient that appears in Equation 3 with respect to all u_i and v_i
2. using your derivation above implement a function that computes the gradient updates given a particular context/input pair and K negative samples.
3. implement a function that samples a context/input from the training data.
   This function should take the context window length, C as a parameter and return a list of context/input word pairs (one for each context word in the
   context window).
   Let us adopt the convention that context windows are always symmetric about the input word and we take C context words on either side of the input word.
   That is, each context window will have 2C context words (you may choose how to deal with words on the edges of the sentence as you wish).
4. implement a function that uses the above function to create a mini-batch of context/input pairs.
5. implement a function that updates the model parameters using the SGD rule in equation 3 given a mini-batch
6. implement a class encapsulating all algorithm hyperparameters including learning rate and mini-batch size
7. implement a function called LearnParamsUsingSGD which takes a training set and hyperparams as input and runs the SGD update on a randomly sampled
   mini-batch at each iteration for a pre-specified number of iterations.
   After each update the parameters should be renormalized so that all vectors are unit length (in terms of the L2 norm).
   After a fixed number of iterations(specified in the hyperparameters) has elapsed the learning rate should be reduced by 50%.
"""

import datetime
import numpy as np
from collections import defaultdict
from functools import partial
import diagnostics

"""
Section 1:
----------
Derive the gradient that appears in Equation 3 with respect to all u_i and v_i:
grad_by_u_t__f(w_t,w_c) = (1 − σ(v_c^T * u_t))*v_c - (σ(v_nk^T * u_t))*v_nk
grad_by_v_c__f = (1 − σ(v_c^T * u_t))*u_t
grad_by_v_nk__f = - (σ(v_nk^T * u_t))*u_t
"""

__sgd_mini_batch_output__ = r"output\sgd_mini_batch_log_L.dat"
__sgd_test_output__ = r"output\sgd_test_log_L.dat"


class SGDHyperParameters:
    """
    Section 6:
    ----------
    Implement a class encapsulating all algorithm hyperparameters including learning rate and mini-batch size.
    """
    def __init__(self,
                 alpha=0.1,
                 eta=10,
                 mini_batch_size=10,
                 X=100):
        """
        Initialization method
        :param alpha: Learning rate
        :param eta: Iteration count for decreasing the learning rate
        :param mini_batch_size: Size of the mini batch of the algorithm
        :param X: Iteration number in the algorithm to calculate mini-batch log-likelihood and full log-likelihood
        """
        self.alpha = alpha
        self.eta = eta
        self.mini_batch_size = mini_batch_size
        self.X = X


class StochasticGradientDescent:
    """
    Implementation of SGD algorithm for training a model
    """
    def __init__(self, hyper_parameters, model):
        """
        Initiazlization method
        :param hyper_parameters: Hyper parameters instance of the SGDHyperParameters class.
        :param model: Instance of the Skip-Gram model.
        """
        self.hyper = hyper_parameters
        self.model = model

    @staticmethod
    def sigmoid(x):
        return 1.0/(1.0+np.exp(-x))

    def get_gradient_updates(self, target_word, context_word, negative_samples, grad_dict_target, grad_dict_context):
        """
        Section 2:
        ----------
        Implement a function that computes the gradient updates given a particular context/input pair and K negative samples.
        for a pair c,t:
            grad_t = grad_t + (1 − σ(v_c^T * u_t))*v_c
            grad_c = grad_c + (1 − σ(v_c^T * u_t))*u_t
            for each non-context-word nk:
                grad_t = grad_t - (σ(v_nk^T * u_t))*v_nk
                grad_nk = grad_nk - (σ(v_nk^T * u_t))*u_t
        :param target_word: Target word from the corpus
        :param context_word: Single context word of the target word
        :param negative_samples: K negative samples of non-context words
        :param grad_dict_target: The gradient dictionary of the target words
        :param grad_dict_context: The gradient dictionary of the context and non-context words
        """
        # TODO: check this and compare to boaz or ido.. !!!
        sigmoid_vcTut = self.sigmoid(np.dot(self.model.u[target_word], self.model.v[context_word]))
        grad_dict_target[target_word] += (1 - sigmoid_vcTut) * (self.model.v[context_word])
        grad_dict_context[context_word] += (1 - sigmoid_vcTut) * (self.model.u[target_word])
        for negative_context_word in negative_samples:
            sigmoid_vnkTut = self.sigmoid(np.dot(self.model.u[target_word], self.model.v[negative_context_word]))
            grad_dict_target[target_word] -= sigmoid_vnkTut * (self.model.v[negative_context_word]) * (1/len(negative_samples)) # TODO: added 1/K fix
            grad_dict_context[negative_context_word] -= sigmoid_vnkTut * (self.model.u[target_word]) * (1/len(negative_samples))  # TODO: added 1/K fix

    @staticmethod
    def sample_word_context_window(dataset, input_word_index, sentence, C, V):
        """
        Section 3:
        ----------
        Implement a function that samples a context/input from the training data. This function should take the context window length, C as a parameter
        and return a list of context/input word pairs (one for each context word in the context window).
        Let us adopt the convention that context windows are always symmetric about the input word and we take C context words on either side of the
        input word.
        That is, each context window will have 2C context words (you may choose how to deal with words on the edges of the sentence as you wish).
        Sample C context words from each size of the input word.
        There can be a situation where the left and right sizes don't have enough words, so we'll chop the unavailable context words from each size.
        :param dataset: Dataset to sample word context from
        :param input_word_index: An input word to randomly select C context words from the left and right.
        :param sentence: The sentence to create the context from
        :param C: The context window size from each side
        :param V: Vocabulary
        :return: List of pairs (context/input) - size 2C.
        """
        # 1. create list of contexts words
        context_words = []
        left_side = sentence[:input_word_index[1]]
        right_side = sentence[input_word_index[1]+1:]
        context_words.extend(list(reversed(left_side))[:C] if len(left_side) > C else left_side)
        context_words.extend(right_side[:C] if len(right_side) > C else right_side)

        # 2. return a list of tuples: [(target_word, context_word1), (target_word, context_word2), (target_word, context_word3)]
        input_context_pairs_list = []
        input_word = dataset[input_word_index[0]][input_word_index[1]]
        for context_word in context_words:
            # before adding the pair to the result, verify that the context word is in the model vocabulary
            if context_word in V:
                tuple_input_context_pair = (input_word, context_word)
                input_context_pairs_list.append(tuple_input_context_pair)
        return input_context_pairs_list

    def create_mini_batch(self, training_set):
        """
        Section 4:
        ----------
        Implement a function that uses the above function to create a mini-batch of context/input pairs.
        :return: Create a mini batch size of context/input pairs from the corpus
        """
        mini_batch_pairs = []
        for index in xrange(self.hyper.mini_batch_size):
            input_word = self.model.sample_target_word()
            sentence = training_set[input_word[0]]
            mini_batch_pairs.append(self.sample_word_context_window(training_set, input_word, sentence, self.model.hyper.C, self.model.V))
        return mini_batch_pairs

    def update_step(self, grad_dict_target, grad_dict_context):
        """
        Section 5:
        ----------
        Implement a function that updates the model parameters using the SGD rule in equation 3 given a mini-batch
        The method only updates the inner values of the vectors.
        """
        # TODO: delete this normalization, and do it after the update step on ALL u,v
        for target_word, gradient in grad_dict_target.items():
            self.model.u[target_word] += self.hyper.alpha * gradient
        for context_word, gradient in grad_dict_context.items():
            self.model.v[context_word] += self.hyper.alpha * gradient

    def create_dataset_target_context_pairs(self, dataset, V):
        """
        Creates a list of pairs
        :param dataset: Dataset list of sentences (corpus)
        :param V: Dataset vocabulary
        :return: The target context pairs from the complete corpus
        """
        dataset_pairs = []
        for sentence_index in xrange(len(dataset)):
            sentence = dataset[sentence_index]
            for word_index in xrange(len(sentence)):
                if sentence[word_index] in self.model.V:
                    dataset_pairs.append(self.sample_word_context_window(dataset, (sentence_index, word_index), sentence, self.model.hyper.C, V))
        return dataset_pairs


def LearnParamsUsingSGD(training_set, hyper_parameters, model, iterations_number, test_set):
    """
    Section 7:
    ----------
    Implement a function called LearnParamsUsingSGD which takes a training set and hyperparams as input and runs the SGD update on a randomly sampled
    mini-batch at each iteration for a pre-specified number of iterations.
    After each update the parameters should be renormalized so that all vectors are unit length (in terms of the L2 norm). After a fixed number of
    iterations (specified in the hyperparameters) has elapsed the learning rate should be reduced by 50%.
    :param training_set: Training data set
    :param hyper_parameters: Hyper parameters of the SGD algorithm
    :param model: Model object
    :param iterations_number: Number of iteration for the SGD algorithm
    :param test_set: Test data set
    :return: Learned SGD algorithm object
    """
    sgd = StochasticGradientDescent(hyper_parameters, model)
    sgd.hyper.iterations_number = iterations_number
    sgd.log_L_train = []
    sgd.log_L_test = []
    T_V, T_unigram, T_probs = diagnostics.create_test_data_structures(model, test_set)
    T_pairs = sgd.create_dataset_target_context_pairs(test_set, T_V)
    with open(__sgd_mini_batch_output__, "w") as sgd_mini_batch, open(__sgd_test_output__, "w") as sgd_test:
        for index in xrange(iterations_number):
            # create mini_batch_pairs = [[(wt1, wt1_c1), (wt1, wt1_c2)], [(wt2, wt2_c1), (wt2, wt2_c2)], [(wt3, wt3_c1), (wt3, wt3_c2)]]
            mini_batch_log_likelihood_parameters = []
            mini_batch_pairs = sgd.create_mini_batch(training_set)
            grad_dict_target = defaultdict(partial(np.zeros, model.hyper.d))
            grad_dict_context = defaultdict(partial(np.zeros, model.hyper.d))

            # Sample a word context pair (wt, wc_list) from dataset: target_context_pairs = [(wt1, wt1_c1), (wt1, wt1_c2)]
            for target_context_pairs in mini_batch_pairs:
                # for (wt,wc) in wc_list: target_word, context_word = wt1, wt1_c1
                for target_word, context_word in target_context_pairs:
                    negative_samples = sgd.model.get_negative_samples(model.V_unigram, model.V_probs)
                    mini_batch_log_likelihood_parameters.append((context_word, target_word, negative_samples))
                    sgd.get_gradient_updates(target_word, context_word, negative_samples, grad_dict_target, grad_dict_context)
            # now update the parameters: u, v AND normalize them according to l2_norm
            sgd.update_step(grad_dict_target, grad_dict_context)
            # normalize all u,v
            for k, v in model.u.items():
                l2_norm = np.linalg.norm(x=v, ord=2, keepdims=True)
                model.u[k] /= l2_norm
            for k, v in model.v.items():
                l2_norm = np.linalg.norm(x=v, ord=2, keepdims=True)
                model.v[k] /= l2_norm
            """
            Section 4.3:
            ------------
            Modify the function LearnParamsUsingSGD from the previous part to print the mini-batch likelihood at each iteration (after the parameter update)
            """
            # compute mini-batch log-likelihood
            mini_batch_log_likelihood = diagnostics.get_log_likelihood_mini_batch(model, mini_batch_log_likelihood_parameters)
            avg_mini_batch_log_L = (mini_batch_log_likelihood / sgd.hyper.mini_batch_size)
            datetime_value = get_timestamp()
            print "{0} - Iteration: {1}, Mini-Batch Log Likelihood: {2} (AVG: {3})".format(datetime_value, index, mini_batch_log_likelihood, avg_mini_batch_log_L)
            sgd.log_L_train.append(avg_mini_batch_log_L)
            sgd_mini_batch.write("{0},{1},{2},{3}\n".format(datetime_value, index, mini_batch_log_likelihood, avg_mini_batch_log_L))

            """
            Section 4.4:
            ------------
            Modify the function LearnParamsUsingSGD to take a second dataset (i.e. a test set).
            Every X iterations, where X is a hyper-parameter the function should save the iteration number and the value of the mini-batch log-likelihood on the 
            train set and the full loglikelihoodon the test sets, given the current settings of the parameters. 
            In order to make the numbers on the same scale report the mean log-likelihood across all examples considered.
            """
            # compute full log-likelihood on the test dataset
            if index % hyper_parameters.X == 0:
                full_log_likelihood = diagnostics.get_log_likelihood(model, T_unigram, T_probs, T_pairs)
                avg_full_log_L = (full_log_likelihood / len(T_pairs))
                datetime_value = get_timestamp()
                print "{0} - Iteration: {1}, Full Log-Likelihood: {2} (AVG: {3})".format(datetime_value, index, full_log_likelihood, avg_full_log_L)
                sgd.log_L_test.append(full_log_likelihood)
                sgd_test.write("{0},{1},{2},{3}\n".format(datetime_value, index, full_log_likelihood, avg_full_log_L))

            # update learning rate if number of iterations passed
            if index % sgd.hyper.eta == 0:
                sgd.hyper.alpha /= 2
    return sgd


def get_timestamp():
    return datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S.%f")
