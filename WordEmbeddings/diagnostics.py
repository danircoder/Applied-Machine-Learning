# coding=utf-8

"""
Since we are maximizing the objective in Equation 2, one might imagine computing the value of this objective every few iterations to see if our
algorithm is learning.
However, this calculation is heavy as there are many elements in the summation. Instead, we can evaluate the ‘mini-batch likelihood’ at each iteration
as a surrogate:

L(Θ) = simga_(w_c,w_i)∈D_mb log P (w_c | w_i)

because each mini-batch is slightly different, these quantities will not be strictly increasing at every iteration.
However, their magnitude should show a trend to larger (negative) numbers.
In this part you are required to:
1. Implement a function that computes the log-likelihood given in Equation 2 which takes the model hyperparameters as input
2. Implement a function that computes the ’mini-batch likelihood’ given in Equation 4
3. Modify the function LearnParamsUsingSGD from the previous part to print the mini-batch likelihood at each iteration (after the parameter update)
4. Modify the function LearnParamsUsingSGD to take a second dataset (i.e. a test set).
   Every X iterations, where X is a hyper-parameter the function should save the iteration number and the value of the mini-batch log-likelihood on
   the train set and the full loglikelihood on the test sets, given the current settings of the parameters.
   In order to make the numbers on the same scale report the mean log-likelihood across all examples considered.
"""


def create_test_data_structures(model, test_dataset):
    # create V - Vocabulary from the training data:
    T_V = model.create_vocabulary(test_dataset)

    # create a subset of the training dataset
    words_to_remove = []
    for key in T_V.iterkeys():
        if key not in model.V:
            words_to_remove.append(key)
    for word in words_to_remove:
        T_V.pop(word)

    # create the vocabulary distribution probabilities and probabilities sum
    T_unigram = model.get_unigram_alpha_dict(T_V)
    T_probs = model.get_vocabulary_probabilities(T_unigram)

    return T_V, T_unigram, T_probs


def get_log_likelihood(model, data_unigram, data_probs, data_pairs):
    """
    Section 1:
    ----------
    Implement a function that computes the log-likelihood given in Equation 2 which takes the model hyperparameters as input
    L(Θ) = sigma_(wc,wi)∈D (log P (wc | wi))
    :param model: Model object
    :param data_unigram: Dataset unigram distribution of the vocabulary
    :param data_probs: Dataset unigram probabilities of the vocabulary
    :param data_pairs: Data set target/context pairs
    :return: Log likelihood of the (wc,wi)∈D
    """
    log_likelihood = 0
    for target_context_pairs in data_pairs:
        for target_word, context_word in target_context_pairs:
            K_words = model.get_negative_samples(data_unigram, data_probs)
            log_likelihood += model.get_log_probability(context_word, target_word, K_words)
    return log_likelihood


def get_log_likelihood_mini_batch(model, mini_batch):
    """
    Section 2:
    ----------
    Implement a function that computes the ’mini-batch likelihood’ given in Equation 4.
    L(Θ) = sigma_(wc,wi)∈Dmb (log P (wc | wi))
    :param model: Model object
    :param mini_batch: The target, context pairs of the mini batch
    :return: Log likelihood of the (wc,wi)∈D_mb
    """
    mini_batch_likelihood = 0
    for context_word, target_word, K_words in mini_batch:
        mini_batch_likelihood += model.get_log_probability(context_word, target_word, K_words)
    return mini_batch_likelihood
