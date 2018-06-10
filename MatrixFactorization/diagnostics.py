# coding=utf-8
"""
Part 7. Diagnostics and Analysis
--------------------------------
In this part you will use the code you implemented above to create several plots (marked by Deliverable tag) that you will turn in along with your code.

In order to make sure our algorithm is implemented correctly we want to see that the training error is being reduced over time in our algorithm.

Deliverable 6.
--------------
Your Code. Be prepared to explain where and how you implemented each of parts 1-6 above.
"""

import evaluation
import algorithm
import model
import time
import numpy as np
import ConfigParser
import plots


__lambda_analysis_output__ = r"output\lambda_analysis_output.dat"
__latent_dimensions_analysis_output__ = r"output\latent_dimensions_analysis_output.dat"
__latent_dimensions_runtime_output__ = r"output\latent_dimensions_runtime_output.dat"

__sgd_train_learning_curve__ = r"output\sgd_train_learning_curve.dat"
__sgd_test_learning_curve__  = r"output\sgd_test_learning_curve.dat"
__als_train_learning_curve__ = r"output\als_train_learning_curve.dat"
__als_test_learning_curve__  = r"output\als_test_learning_curve.dat"


def deliverable1():
    """
    Deliverable 1.
    --------------
    Plot the train and test error as a function of training iteration. Train and test plots should appear in the same figure, clearly marked.
    Each algorithm (ALS and SGD) should have a separate plot. You may use whatever configuration of hyperparameters you wish for this plot, but make sure you
    specify the choice.
    Hyperparameters can sometimes have a great effect on generalization performance. We will consider the effect of regularization and the number of dimensions.
    """
    # Plot according to the LearnModelFromDataUsingALS and LearnModelFromDataUsingSGD outputs to file from the main flow
    deliverable1_plot(__sgd_train_learning_curve__, __sgd_test_learning_curve__, __als_train_learning_curve__,
                          __als_test_learning_curve__)


def deliverable2(ratings_train, ratings_test):
    """
    Deliverable 2.
    --------------
    For one of the two algorithms fix all hyper-parameters except for the regularization parameters. Make the simplifying modeling choice
    λv = λu = λbu = λbv = λ (i.e. all the regularization parameters are set to the same value). Vary the value of λ from 0.1 to 1000 in 3 multiples of 10.
    Choose two of the generalization metrics on the test set output by your code and plot the results as a function of log λ. All hyper-parameter configurations
    of the algorithm should be clearly specified.
    """
    min_lambda = 0.1
    max_lambda = 1000
    lambda_multiplier = 10
    lambda_value = min_lambda
    with open(__lambda_analysis_output__, 'w') as lambda_analysis:
        print "Lambda Analysis:"
        while lambda_value <= max_lambda:
            hyper = algorithm.SGDHyperParameters(alpha=0.01,
                                                 eta=25,
                                                 lambda_user=lambda_value,
                                                 lambda_item=lambda_value,
                                                 lambda_bias_user=lambda_value,
                                                 lambda_bias_item=lambda_value,
                                                 latent_factor=20)
            model_object = model.MFModel(ratings_train, hyper)
            model_object = algorithm.LearnModelFromDataUsingSGD(ratings_train, model_object, hyper)
            ranked_items_list = evaluation.get_ranked_items_list(model_object, ratings_test)
            rmse = evaluation.root_mean_squared_error(ranked_items_list, ratings_test)
            p_at_k = evaluation.precision_at_K(ranked_items_list, ratings_test, 5)
            print "Lambda: {0}, log(Lambda): {1}, RMSE: {2}, Precision@K: {3}".format(lambda_value, np.log10(lambda_value), rmse, p_at_k)
            lambda_analysis.write("{0},{1},{2},{3}\n".format(lambda_value, np.log10(lambda_value), rmse, p_at_k))
            lambda_value *= lambda_multiplier
    plots.deliverable2_plot(__lambda_analysis_output__)


def deliverable3(ratings_train, ratings_test):
    """
    Deliverable 3.
    --------------
    For the same algorithm you picked above and for (one of) the best setting(s) of λ based on your analysis above do the following: fix all hyper-parameters
    except for d the number of latent dimensions. Now run the flow several times varying d for all values in the set {2, 4, 10, 20, 40, 50, 70, 100, 200}.
    Plot the values of your choice of generalization metrics (same choice as above). Again specify all your modeling choices clearly Some hyper-parameters can
    also significantly effect the training time of the algorithm.
    """
    latent_dimensions = [2, 4, 10, 20, 40, 50, 70, 100, 200]
    with open(__latent_dimensions_analysis_output__, 'w') as dimensions_analysis:
        print "Dimensions Analysis:"
        for latent_dim in latent_dimensions:
            hyper = algorithm.SGDHyperParameters(alpha=0.01,
                                                 eta=25,
                                                 lambda_user=0.1,
                                                 lambda_item=0.1,
                                                 lambda_bias_user=0.1,
                                                 lambda_bias_item=0.1,
                                                 latent_factor=latent_dim)
            model_object = model.MFModel(ratings_train, hyper)
            model_object = algorithm.LearnModelFromDataUsingSGD(ratings_train, model_object, hyper)
            ranked_items_list = evaluation.get_ranked_items_list(model_object, ratings_test)
            rmse = evaluation.root_mean_squared_error(ranked_items_list, ratings_test)
            p_at_k = evaluation.precision_at_K(ranked_items_list, ratings_test, 5)
            print "Dimension: {0}, RMSE: {1}, Precision@K: {2}".format(latent_dim, rmse, p_at_k)
            dimensions_analysis.write("{0},{1},{2}\n".format(latent_dim, rmse, p_at_k))
    plots.deliverable3_plot(__latent_dimensions_analysis_output__)


def deliverable4(ratings_train):
    """
    Deliverable 4.
    --------------
    For the same experiments as above plot the training run-time as a function of d.
    """
    latent_dimensions = [2, 4, 10, 20, 40, 50, 70, 100, 200]
    with open(__latent_dimensions_analysis_output__, 'w') as dimensions_analysis:
        print "Dimensions Runtime:"
        for latent_dim in latent_dimensions:
            hyper = algorithm.SGDHyperParameters(alpha=0.01,
                                                 eta=25,
                                                 lambda_user=0.1,
                                                 lambda_item=0.1,
                                                 lambda_bias_user=0.1,
                                                 lambda_bias_item=0.1,
                                                 latent_factor=latent_dim)
            model_object = model.MFModel(ratings_train, hyper)
            timer = time.clock()
            algorithm.LearnModelFromDataUsingSGD(ratings_train, model_object, hyper)
            training_time = time.clock() - timer
            print "Dimension: {0},Training Time: {1}".format(latent_dim, training_time)
            dimensions_analysis.write("{0},{1}\n".format(latent_dim, training_time))
    plots.deliverable4_plot(__latent_dimensions_analysis_output__)


def deliverable5(ratings_train, movies_dict):
    """
    Deliverable 5.
    --------------
    For 5 users who have rated 3 or more items in the training set, provide a human readable format of the recommendations that would result from
    applying the learned model.
    """
    selected_users_indexes = []
    users_list = xrange(ratings_train.shape[0])
    while len(selected_users_indexes) < 5:
        random_user = np.random.choice(users_list, 1)[0]
        if len(ratings_train[random_user].nonzero()[0]) >= 3:
            selected_users_indexes.append(random_user)

    config = ConfigParser.RawConfigParser()
    config.read('config.ini')
    hyper = algorithm.SGDHyperParameters(alpha=config.getfloat("SGD", "LearningRate"),
                                         eta=config.getint("SGD", "Epochs"),
                                         lambda_user=config.getfloat("SGD", "LambdaUser"),
                                         lambda_item=config.getfloat("SGD", "LambdaItem"),
                                         lambda_bias_user=config.getfloat("SGD", "LambdaUserBias"),
                                         lambda_bias_item=config.getfloat("SGD", "LambdaItemBias"),
                                         latent_factor=config.getint("SGD", "LatentDimensions"))
    model_object = model.MFModel(ratings_train, hyper)
    model_object = algorithm.LearnModelFromDataUsingSGD(ratings_train, model_object, hyper)
    predictions = evaluation.get_ranked_items_list(model_object, ratings_train)

    k = 5
    user_id_len = 6
    movie_id_len = 7
    movie_title_len = 50
    movie_genre_len = 75
    prediction_len = 12
    ground_truth_len = 12

    for user in selected_users_indexes:
        print "Displaying Top {0} Recommended Movies for User {1}:".format(k, user)
        top_user_predictions = predictions[user][:k]
        print "+{0}+{1}+{2}+{3}+{4}+{5}+".format("-" * user_id_len, "-" * movie_id_len, "-" * movie_title_len, "-" * movie_genre_len, "-" * prediction_len, "-" * ground_truth_len)
        print "| {0} | {1} | {2} | {3} | {4} | {5} |".format("User".ljust(user_id_len - 2),
                                                             "Movie".ljust(movie_id_len - 2),
                                                             "Title".ljust(movie_title_len - 2),
                                                             "Genres".ljust(movie_genre_len - 2),
                                                             "Prediction".ljust(prediction_len - 2),
                                                             "True Label".ljust(ground_truth_len - 2))
        print "+{0}+{1}+{2}+{3}+{4}+{5}+".format("-" * user_id_len, "-" * movie_id_len, "-" * movie_title_len, "-" * movie_genre_len, "-" * prediction_len, "-" * ground_truth_len)
        for movie, prediction in top_user_predictions:
            print "| {0} | {1} | {2} | {3} | {4} | {5} |".format(str(user).ljust(user_id_len - 2),
                                                                 str(movie).ljust(movie_id_len - 2),
                                                                 movies_dict[movie][0].ljust(movie_title_len - 2),
                                                                 movies_dict[movie][1].ljust(movie_genre_len - 2),
                                                                 str("%.7f" % prediction).ljust(prediction_len - 2),
                                                                 str(ratings_train[user, movie]).ljust(ground_truth_len - 2))
        print "+{0}+{1}+{2}+{3}+{4}+{5}+".format("-" * user_id_len, "-" * movie_id_len, "-" * movie_title_len, "-" * movie_genre_len, "-" * prediction_len, "-" * ground_truth_len)
