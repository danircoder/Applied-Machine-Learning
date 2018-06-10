# coding=utf-8

"""
There are few neat ways to evaluate word2vec models beyond measuring log-likelihood.
Some ideas include:

• predicting the most likely context given an input word:
  wˆc = arg_max_w_c P (w_c | w_i)

• predicting the input word given a particular string of words, w_1, ..., w_N :
  wˆi = arg_max_w_i P (w_1, ... , w_i−C, ..., wi, ..., w_i+C, ..., w_N ) = arg_max_w_i pi_c=−C^C P (w_i+c | w_i)

• Solving ‘SAT analogy questions’, for example “man is to woman as king is to ?” using linear relations on the word embeddings in the model as follows:
  solution to analogy = arg_max_i u_i^T (u_man − u_woman + u_king)

In this part you are required to:
1. Implement a function that takes in a model and an input word and outputs the 10 most likely contexts.
2. Implement a function that takes in a model and a set of context words and outputs the 10 most likely inputs
3. Implement a function that takes a list of input words and visualizes the words on a scatter plot using the first 2 elements of each word’s
   embedding vector.
4. implement an ‘analogy solver’ function which takes the first three parts of an analogy and outputs the top 10 results (along with their score)
   for the last part.
"""

import numpy as np
import matplotlib.pyplot as plt


def predict_contexts(model, target_word):
    """
    Section 1:
    ----------
    Implement a function that takes in a model and an input word and outputs the 10 most likely contexts.
    w^c = arg max P (wc j wi)
    :param model: Model object
    :param target_word: Target word to predict context
    :return: 10 most likely contexts
    """
    contexts_prob_dict = {}
    K_words = model.get_negative_samples(model.V_unigram, model.V_probs)
    for w_c in model.V.keys():
        contexts_prob_dict[w_c] = np.exp(model.get_log_probability(w_c=w_c, w_t=target_word, K_words=K_words))

    top_10_likely_contexts = []
    print "---for input target_word: {0}, Top 10 most likely contexts are:".format(target_word)
    for key, value in sorted(contexts_prob_dict.iteritems(), key=lambda (k, v): (v, k), reverse=True)[:10]:
        print "----word: {}\t\t\t\tprobability: {}".format(key, value)
        top_10_likely_contexts.append(key)
    return top_10_likely_contexts


def predict_target(model, context_words):
    """
    Section 2:
    ----------
    Implement a function that takes in a model and a set of context words and outputs the 10 most likely inputs
    :param model: Model object
    :param context_words: Set of context words to predict target word
    :return: 10 most likely target words
    """
    contexts_prob_dict = dict((word, 1) for word in model.V.keys())
    K_words = model.get_negative_samples(model.V_unigram, model.V_probs)
    for w_t in model.V.keys():
        for w_c in context_words:
            contexts_prob_dict[w_t] *= np.exp(model.get_log_probability(w_c=w_c, w_t=w_t, K_words=K_words))

    top_10_likely_targets = []
    print "---for input context_words: {0}, Top 10 most likely contexts are:".format(context_words)
    for key, value in sorted(contexts_prob_dict.iteritems(), key=lambda (k, v): (v, k), reverse=True)[:10]:
        print "----word: {}\t\t\t\tprobability: {}".format(key, value)
        top_10_likely_targets.append(key)
    return top_10_likely_targets


def scatter_plot_words_vector(model, words, parameters, output_path):
    """
    Section 3:
    ----------
    Implement a function that takes a list of input words and visualizes the words on a scatter plot using the first 2 elements of each word’s embedding vector.
    :param output_path:
    :param parameters:
    :param model: Model object
    :param words: List of words to scatter plot
    """
    vectors_array = np.empty((0, 2), dtype='f')
    words_array = []
    for word in words:
        if word in model.u.keys():
            words_array.append(word)
            vectors_array = np.append(vectors_array, np.array([(model.u[word][0:2] + model.v[word][0:2]) / 2.0]), axis=0)
    x_coords = vectors_array[:, 0]
    y_coords = vectors_array[:, 1]
    plt.scatter(x_coords, y_coords)
    for label, x, y in zip(words_array, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min() + 0.00005, x_coords.max() + 0.00005)
    plt.ylim(y_coords.min() + 0.00005, y_coords.max() + 0.00005)
    _title = unicode('Visualization of words on a scatter plot. \nParameters: ') + parameters.decode('utf-8')
    plt.title(_title)
    plt.savefig(output_path, bbox_inches='tight')


def analogy_solver(model, first_word, second_word, third_word):
    """
    Section 4:
    ----------
    implement an ‘analogy solver’ function which takes the first three parts of an analogy and outputs the top 10 results (along with their score) for the
    last part.
    :param model: Model object
    :param first_word: First word of the analogy
    :param second_word: Second word of the analogy
    :param third_word: Third word of the analogy
    """
    # TODO: we can 3 context words, get the top 10 target words !!! also return the score of the target words.
    # TODO: print the words and their score
    last_word_prob_dict = {}
    fixed_part = model.u[first_word] - model.u[second_word] + model.u[third_word]
    for u_i in model.V.keys():
        last_word_prob_dict[u_i] = model.u[u_i].T.dot(fixed_part)

    top_10_likely_last_word = []
    print "---for input words: {0} is to {1}  as  {2} is to ___\nTop 10 most likely contexts are:".format(first_word, second_word, third_word)
    for key, value in sorted(last_word_prob_dict.iteritems(), key=lambda (k, v): (v, k), reverse=True)[:10]:
        print "----word: {}\t\t\t\tscore: {}".format(key, value)
        top_10_likely_last_word.append(key)
    return top_10_likely_last_word


def analogy_solver_context_embeddings(model, first_word, second_word, third_word):
    """
    Section 4:
    ----------
    implement an ‘analogy solver’ function which takes the first three parts of an analogy and outputs the top 10 results (along with their score) for the
    last part.
    :param model: Model object
    :param first_word: First word of the analogy
    :param second_word: Second word of the analogy
    :param third_word: Third word of the analogy
    """
    # TODO: we can 3 context words, get the top 10 target words !!! also return the score of the target words.
    # TODO: print the words and their score
    last_word_prob_dict = {}
    fixed_part = model.v[first_word] - model.v[second_word] + model.v[third_word]
    for u_i in model.V.keys():
        last_word_prob_dict[u_i] = model.v[u_i].T.dot(fixed_part)

    top_10_likely_last_word = []
    print "---for input words: {0} is to {1}  as  {2} is to ___\nTop 10 most likely contexts are:".format(first_word, second_word, third_word)
    for key, value in sorted(last_word_prob_dict.iteritems(), key=lambda (k, v): (v, k), reverse=True)[:10]:
        print "----word: {}\t\t\t\tprobability: {}".format(key, value)
        top_10_likely_last_word.append(key)
    return top_10_likely_last_word
