# coding=utf-8
"""
Part 5. Evaluation
------------------
In class we discussed a number of evaluation metrics that can be used to assess the performance of your approach:
• Root Mean Squared Error (RMSE)
• Mean Percentile Rank (MPR)
• Precision At K (P@k)
• Recall At K (R@k)
• Mean Average Precision(MAP)
"""

import numpy as np
import itertools


def __get_true_positives(ranked_list, ground_truth, k):
    true_positives = 0
    # sort ranked items by rank (low to high) and select last k and reverse the list
    # top_k_item_indexes = ranked_list.argsort()[-k:][::-1]
    top_k_item_indexes = ranked_list[:k]
    # get none ranked items from the ground truth
    ground_truth = ground_truth.nonzero()[0]
    for item_index, prediction in top_k_item_indexes:
        if item_index in ground_truth:
            true_positives += 1
    return true_positives


def __user_precision_at_k(ranked_list, ground_truth, k):
    true_positives = __get_true_positives(ranked_list, ground_truth, k)
    return float(true_positives) / float(k)


def __user_recall_at_k(ranked_list, ground_truth, k):
    true_positives = __get_true_positives(ranked_list, ground_truth, k)
    N = len(ground_truth.nonzero()[0])
    return float(true_positives) / float(N)


def __user_average_precision(ranked_list, ground_truth, k):
    avg_precision = 0
    previous_recall = 0
    for k_iter in xrange(1, k+1):
        p_at_k = __user_precision_at_k(ranked_list, ground_truth, k_iter)
        r_at_k = __user_recall_at_k(ranked_list, ground_truth, k_iter)
        avg_precision += (p_at_k * (r_at_k - previous_recall))
        previous_recall = r_at_k
    return avg_precision


def get_ranked_items_list(model, ratings):
    """
    Predict
    :param model: Learned model object
    :param ratings: Test set ratings
    :return: Sorted (high to low) ranked items list for each user (matrix)
    """
    users_count, items_count = ratings.shape
    ranked_items_list = np.zeros(ratings.shape, dtype=[('item', int), ('pred', float)])
    for user in xrange(users_count):
        for item in xrange(items_count):
            ranked_items_list[user, item] = (item, model.predict(user, item))
        ranked_items_list[user][::-1].sort(order='pred')
    return ranked_items_list


def root_mean_squared_error(ranked_items_list, ratings):
    """
    Calculate the RMSE value for the given model and ratings matrix parameters.
    Formula:
    --------
    RMSE = sqrt(1/|T| * sigma_(m,n)∈I (rm,n − rˆm,n)^2)
    :param ranked_items_list: Predicted ranked items list for each user
    :param ratings: Ratings matrix to check RMSE against
    :return: The RMSE value
    """
    error = 0
    for user, item in itertools.product(xrange(ranked_items_list.shape[0]), xrange(ranked_items_list.shape[1])):
        error += np.power(ratings[user, item] - ranked_items_list[user, item][1], 2)
    error /= float(ratings.shape[0] * ratings.shape[1])
    rmse = np.sqrt(error)
    return rmse


def mean_percentile_rank(ranked_items_list, ratings):
    """
    Calculate the MPR value for the given model and ratings matrix parameters.
    Formula:
    --------
    MPR = 1/|T| * sigma_u∈T (ARank(u)/|catalog|)

    T - our test dataset of user ground truth
    ARank(u) - the average rank of held out items for test user u
    |catalog| - size of item catalog
    :param ranked_items_list: Predicted ranked items list for each user
    :param ratings: Ratings matrix to check MPR against
    :return: The MPR value
    """
    percentile_rank = 0
    user_count = ranked_items_list.shape[0]
    item_count = ranked_items_list.shape[1]
    for user_index in xrange(user_count):
        user_item_indexes = ratings[user_index].nonzero()[0]
        user_percentile_rank = 0
        for user_item_index in user_item_indexes:
            for item_index in xrange(item_count):
                if user_item_index == ranked_items_list[user_index, item_index][0]:
                    user_percentile_rank += float(item_index + 1) / float(item_count)
                    break
        # user_percentile_rank /= len(user_item_indexes)
        percentile_rank += user_percentile_rank
    percentile_rank /= float(user_count)
    return percentile_rank


def precision_at_K(ranked_items_list, ratings, k):
    """
    Calculate the P@k value for the given model and ratings matrix parameters.
    Formula:
    --------
    P@k = TP / (TP + FP) = TP / k

    TP - number of results in the top k and ground truth
    FP - number of results in top k but not in ground truth
    :param ranked_items_list: Predicted ranked items list for each user
    :param ratings: Ratings matrix to check P@k against
    :param k: Items count
    :return: The P@k value
    """
    precision = 0
    user_count = ranked_items_list.shape[0]
    for user_index in xrange(user_count):
        precision += __user_precision_at_k(ranked_items_list[user_index], ratings[user_index], k)
    precision /= float(user_count)
    return precision


def recall_at_K(ranked_items_list, ratings, k):
    """
    Calculate the R@k value for the given model and ratings matrix parameters.
    Formula:
    --------
    R@k = TP / (TP + TN) = TP / N

    TP - number of results in the top k and ground truth
    TN - number of results in ground truth but not top k
    N - total number of results in ground truth
    :param ranked_items_list: Predicted ranked items list for each user
    :param ratings: Ratings matrix to check R@k against
    :param k: Items count
    :return: The R@k value
    """
    true_positive = 0
    user_count = ranked_items_list.shape[0]
    N = len(np.nonzero(ratings)[0])
    for user_index in xrange(user_count):
        # top_k_indexes = ranked_items_list[user_index].argsort()[-k:][::-1]
        top_k_indexes = ranked_items_list[user_index][:k]
        ground_truth = ratings[user_index].nonzero()[0]
        for item_index, prediction in top_k_indexes:
            if item_index in ground_truth and true_positive < N:
                true_positive += 1
    recall = float(true_positive) / float(N)
    return recall


def mean_average_precision(ranked_items_list, ratings):
    """
    Calculate the MAP value for the given model and ratings matrix parameters.
    Formula:
    --------
    AvgPrecision_user = sigma_k=1 (P_u@k * (R_u@k − R_u@(k − 1))
    MAP = 1/|T| * sigma_u∈T (AvgPrecision_user)

    T - our test dataset of user ground truth
    P_u@k - Precision@k for test user u
    R_u@k - Recall@k for test user u
    :param ranked_items_list: Predicted ranked items list for each user
    :param ratings: Ratings matrix to check MAP against
    :return: The MAP value
    """
    sum_user_avg_precision = 0
    user_count = ranked_items_list.shape[0]
    for user_index in xrange(user_count):
        user_item_count = len(ratings[user_index].nonzero()[0])
        sum_user_avg_precision += __user_average_precision(ranked_items_list[user_index], ratings[user_index], user_item_count)
    mean_avg_precision = float(sum_user_avg_precision) / float(user_count)
    return mean_avg_precision
